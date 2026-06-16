import tqdm.auto as tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from unityagents import UnityEnvironment
from actor import Actor
from critic import Critic
from replay_buffer import ReplayBuffer
import torch.optim as optim
from collections import deque
from typing import Deque, Optional, Dict

BEST_ACTOR_PATH = "best_actor.pth"
BEST_CRITIC_PATH = "best_critic.pth"
OUTPUT_DIM = 4


class AgentHarness:

    num_agents: int
    actor: Actor
    critic: Critic
    target_actor: Actor
    target_critic: Critic
    replay_buffer: ReplayBuffer

    discount_factor: float
    ENV_STATE_DIM = 33

    def __init__(
        self,
        replay_buffer_max_size: int,
        score_window_max_size: int,
        discount_factor: float,
        actor_lr: float,
        critic_lr: float,
        replay_buffer_sample_size: int,
        actor_action_l2_coef: float = 1e-3,
        policy_delay: int = 2,
        load_best: bool = False,
    ) -> None:
        self.num_agents = 1

        self.actor = Actor(input_dim=self.ENV_STATE_DIM, output_dim=OUTPUT_DIM)
        self.target_actor = Actor(input_dim=self.ENV_STATE_DIM, output_dim=OUTPUT_DIM)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(observation_dim=self.ENV_STATE_DIM, action_dim=OUTPUT_DIM)
        self.target_critic = Critic(
            observation_dim=self.ENV_STATE_DIM, action_dim=OUTPUT_DIM
        )
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        if load_best:
            self.actor.load_state_dict(torch.load(BEST_ACTOR_PATH))
            self.critic.load_state_dict(torch.load(BEST_CRITIC_PATH))

        # Initialize target networks with the same weights as the main networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.replay_buffer_sample_size = replay_buffer_sample_size

        self.replay_buffer = ReplayBuffer(max_size=replay_buffer_max_size)
        self.score_window_max_size = score_window_max_size
        self.discount_factor = discount_factor
        self.actor_action_l2_coef = actor_action_l2_coef
        self.policy_delay = policy_delay
        self.last_action_clip_fraction = 0.0
        self.last_policy_saturation_fraction = 0.0

    def act(self, states: np.ndarray, noise_scale: float = 0.1) -> np.ndarray:
        """
        Select an action for each agent.
        """
        torch_states = torch.from_numpy(states).float()

        with torch.no_grad():
            actions = self.actor(torch_states).cpu().numpy()

        # How often the policy itself is near action bounds before exploration noise.
        self.last_policy_saturation_fraction = float(np.mean(np.abs(actions) > 0.95))

        noisy_actions = actions + noise_scale * np.random.randn(*actions.shape)
        self.last_action_clip_fraction = float(np.mean(np.abs(noisy_actions) > 1.0))

        # all actions between -1 and 1
        return np.clip(noisy_actions, -1, 1)

    def rollout(
        self,
        env: UnityEnvironment,
        num_episodes: int,
        max_train_steps_per_episode: int = 100,
        exit_on_solve: bool = False,
        noise_decay: float = 0.995,
    ) -> Deque[float]:
        """
        Generate experience by interacting with the environment and store it in the replay buffer
        for the specified number of episodes.
        """
        noise_scale = 0.15
        noise_min = 0.005
        score_window: Deque[float] = deque()
        all_scores: Deque[float] = deque()
        warmup_episodes = 5
        best_avg_score = 0.0
        avg_score = 0.0
        total_steps = 0
        update_after = 1000  # Start training after 1000 steps collected

        with tqdm.trange(num_episodes, desc="Rollout") as pbar:

            for ep_num in pbar:
                brain_name = env.brain_names[0]
                # reset the environment
                env_info = env.reset(train_mode=True)[brain_name]
                scores = np.zeros(
                    self.num_agents
                )  # initialize the score (for each agent)
                # get the current state (for each agent)
                states = env_info.vector_observations
                episode_steps = 0
                action_mag_sum = 0.0
                clip_fraction_sum = 0.0
                clip_fraction_steps = 0
                policy_sat_sum = 0.0
                policy_sat_steps = 0

                while True:
                    if ep_num < warmup_episodes:
                        actions = np.random.uniform(
                            -1, 1, (self.num_agents, OUTPUT_DIM)
                        )
                    else:
                        actions = self.act(
                            states, noise_scale=noise_scale
                        )  # select an action (for each agent)
                        clip_fraction_sum += self.last_action_clip_fraction
                        clip_fraction_steps += 1
                        policy_sat_sum += self.last_policy_saturation_fraction
                        policy_sat_steps += 1

                    action_mag_sum += float(np.mean(np.abs(actions)))
                    # send all actions to tne environment
                    env_info = env.step(actions)[brain_name]

                    # get next state (for each agent)
                    next_states = env_info.vector_observations
                    rewards = env_info.rewards  # get reward (for each agent)
                    dones = env_info.local_done  # see if episode finished

                    for i in range(self.num_agents):
                        self.replay_buffer.add_entry(
                            obs=states[i],
                            acts=actions[i],
                            rewards=rewards[i],
                            next_obs=next_states[i],
                            done_flags=dones[i],
                        )

                    scores += rewards  # update the score (for each agent)
                    states = next_states  # roll over states to next time step
                    episode_steps += 1

                    if np.any(dones):  # exit loop if episode finished
                        break

                # Track total steps and train after warmup
                total_steps += episode_steps
                if total_steps >= update_after:
                    # Train proportionally to episode length, capped at max
                    train_metrics = self.train(
                        num_steps=min(episode_steps, max_train_steps_per_episode)
                    )
                else:
                    train_metrics = {"actor_loss": None, "critic_loss": None}

                score = np.max(scores)
                score_window.append(score)
                all_scores.append(float(score))

                if len(score_window) > self.score_window_max_size:
                    score_window.popleft()

                avg_score = np.mean(score_window)

                if avg_score >= 30.0:
                    pbar.write(
                        f"Environment solved at episode {ep_num + 1} with avg score {avg_score:.2f}!"
                    )
                    if exit_on_solve:
                        break

                if avg_score > best_avg_score:
                    best_avg_score = avg_score
                    torch.save(self.actor.state_dict(), BEST_ACTOR_PATH)
                    torch.save(self.critic.state_dict(), BEST_CRITIC_PATH)

                mean_abs_action = action_mag_sum / max(episode_steps, 1)
                mean_clip_fraction = clip_fraction_sum / max(clip_fraction_steps, 1)
                mean_policy_sat = policy_sat_sum / max(policy_sat_steps, 1)

                # Adapt noise to keep clipping at a useful but not destructive level.
                if mean_clip_fraction > 0.20:
                    noise_scale = max(noise_min, noise_scale * 0.90)
                elif mean_clip_fraction < 0.05:
                    noise_scale = max(noise_min, noise_scale * 0.995)
                else:
                    noise_scale = max(noise_min, noise_scale * noise_decay)
                avg_score = np.mean(score_window)

                def fmt_metric(value: Optional[float]) -> str:
                    return "n/a" if value is None else f"{value:.3f}"

                pbar.set_postfix(
                    {
                        "Avg100": f"{avg_score:.2f}",
                        "Noise": f"{noise_scale:.3f}",
                        "CritLoss": fmt_metric(train_metrics["critic_loss"]),
                        "ActLoss": fmt_metric(train_metrics["actor_loss"]),
                        "|a|": f"{mean_abs_action:.3f}",
                        "Clip%": f"{100.0 * mean_clip_fraction:.1f}",
                        "PolSat%": f"{100.0 * mean_policy_sat:.1f}",
                    }
                )

        return all_scores

    def train(self, num_steps: int) -> Dict[str, Optional[float]]:
        if len(self.replay_buffer) < self.replay_buffer_sample_size:
            return {"actor_loss": None, "critic_loss": None}

        actor_losses: list[float] = []
        critic_losses: list[float] = []

        for step in range(num_steps):
            buffer_samples = self.replay_buffer.sample(
                batch_size=self.replay_buffer_sample_size
            )

            # Next action based on next state, predicted by target actor network
            with torch.no_grad():
                target_actor_action = self.target_actor(buffer_samples["next_obs"])

                # Q targets for current states (y_i)
                output_values = buffer_samples["rewards"].unsqueeze(
                    1
                ) + self.discount_factor * self.target_critic(
                    buffer_samples["next_obs"], target_actor_action
                ) * (
                    1 - buffer_samples["done_flags"].unsqueeze(1)
                )

            # Get critic loss
            expected_output = self.critic(buffer_samples["obs"], buffer_samples["acts"])
            critic_loss = F.mse_loss(expected_output, output_values)
            # Minimize the loss
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.critic_optimizer.step()
            critic_losses.append(float(critic_loss.item()))

            # Delay policy updates to reduce overfitting to critic errors.
            if step % self.policy_delay == 0:
                actor_prediction = self.actor(buffer_samples["obs"])
                actor_q_loss = -self.critic(
                    buffer_samples["obs"], actor_prediction
                ).mean()
                # Keep actions away from permanent +/-1 saturation.
                action_reg = self.actor_action_l2_coef * actor_prediction.pow(2).mean()
                actor_loss = actor_q_loss + action_reg

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                self.actor_optimizer.step()
                actor_losses.append(float(actor_loss.item()))

                # Soft updates after policy update.
                self.soft_update(self.actor, self.target_actor)
                self.soft_update(self.critic, self.target_critic)

        return {
            "actor_loss": (
                float(np.mean(actor_losses)) if len(actor_losses) > 0 else None
            ),
            "critic_loss": (
                float(np.mean(critic_losses)) if len(critic_losses) > 0 else None
            ),
        }

    def soft_update(
        self, net: nn.Module, target_net: nn.Module, interp_factor: float = 0.001
    ) -> None:

        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.lerp_(param.data, interp_factor)

    def test(self, env: UnityEnvironment, num_episodes: int) -> None:
        brain_name = env.brain_names[0]
        # brain = env.brains[brain_name]

        for _ in tqdm.trange(num_episodes):
            env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
            # get the current state (for each agent)
            states = env_info.vector_observations
            scores = np.zeros(self.num_agents)  # initialize the score (for each agent)
            while True:
                actions = self.act(states)  # select an action (for each agent)
                # send all actions to tne environment
                env_info = env.step(actions)[brain_name]

                # get next state (for each agent)
                next_states = env_info.vector_observations
                dones = env_info.local_done  # see if episode finished
                scores += env_info.rewards  # update the score (for each agent)
                states = next_states  # roll over states to next time step

                if np.any(dones):  # exit loop if episode finished
                    break


def main() -> None:
    path = R"C:\Users\eliot\Documents\GitHub\deep_rl_project_2\Reacher_Windows_x86_64\Reacher.exe"
    env = UnityEnvironment(file_name=path, worker_id=1)

    AgentHarness(
        replay_buffer_max_size=100_000,
        score_window_max_size=100,
        discount_factor=0.99,
        actor_lr=3e-5,
        critic_lr=3e-4,
        replay_buffer_sample_size=100,
        actor_action_l2_coef=1e-3,
        policy_delay=2,
    ).rollout(env, num_episodes=10_000)


if __name__ == "__main__":
    main()
