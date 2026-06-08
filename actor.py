import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    layer_1: nn.Linear
    layer_2: nn.Linear
    layer_3: nn.Linear

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256) -> None:
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, output_dim)

        # Small uniform init on final layer to keep initial actions near zero,
        # preventing saturated tanh outputs early in training (per DDPG paper).
        nn.init.uniform_(self.layer_3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.layer_3.bias, -3e-3, 3e-3)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Feed the observation through the neural network.
        """

        x_1 = F.relu(self.ln1(self.layer_1(obs)))
        x_2 = F.relu(self.layer_2(x_1))
        # Normalize output to closed interval [-1, 1]
        return torch.tanh(self.layer_3(x_2))
