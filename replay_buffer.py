import numpy as np
from collections import deque
from typing import NamedTuple, Deque

import random
import torch


class ReplayBufferEntry(NamedTuple):
    obs: np.ndarray
    acts: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    done_flags: np.ndarray


class ReplayBuffer:
    buffer: Deque[ReplayBufferEntry]

    def __init__(self, max_size: int = 1_000_000) -> None:
        self.buffer = deque(maxlen=max_size)

    def __len__(self) -> int:
        return len(self.buffer)

    def add_entry(
        self,
        obs: np.ndarray,
        acts: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        done_flags: np.ndarray,
    ) -> None:
        entry = ReplayBufferEntry(obs, acts, rewards, next_obs, done_flags)
        self.buffer.append(entry)

    def sample(self, batch_size: int) -> dict:
        obs_batch_list = []
        acts_batch_list = []
        rewards_batch_list = []
        next_obs_batch_list = []
        done_flags_batch_list = []

        for sample in random.sample(self.buffer, batch_size):
            obs_batch_list.append(sample.obs)
            acts_batch_list.append(sample.acts)
            rewards_batch_list.append(sample.rewards)
            next_obs_batch_list.append(sample.next_obs)
            done_flags_batch_list.append(sample.done_flags)

        return dict(
            obs=torch.Tensor(np.array(obs_batch_list)),
            acts=torch.Tensor(np.array(acts_batch_list)),
            rewards=torch.Tensor(np.array(rewards_batch_list)),
            next_obs=torch.Tensor(np.array(next_obs_batch_list)),
            done_flags=torch.Tensor(np.array(done_flags_batch_list)),
        )
