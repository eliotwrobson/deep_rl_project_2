import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    layer_1: nn.Linear
    layer_2: nn.Linear
    layer_3: nn.Linear

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        """
        The input dimension is scaled by the number of individual agents since we
        have a unified critic.
        """
        super(Critic, self).__init__()

        self.layer_1 = nn.Linear(observation_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, 1)

        # Small uniform init on final layer to keep initial Q-values near zero,
        # stabilizing early TD targets (per DDPG paper).
        nn.init.uniform_(self.layer_3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.layer_3.bias, -3e-3, 3e-3)

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Feed the observation through the neural network.
        """

        x_1 = F.relu(self.bn1(self.layer_1(observations)))
        x_2 = torch.cat((x_1, actions), dim=1)
        x_3 = F.relu(self.layer_2(x_2))
        # No need to normalize value of Q table.
        return self.layer_3(x_3)
