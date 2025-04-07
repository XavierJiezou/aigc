from torch import nn as nn
import torch


class Expert(nn.Module):
    def __init__(self, input_dim):
        super(Expert, self).__init__()
        hidden_dim = input_dim // 2
        self.net = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, input_dim, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


class Gating(nn.Module):
    def __init__(self, input_dim, num_experts=20):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Conv2d(input_dim, num_experts, kernel_size=1)

    def forward(self, x):
        weights = self.gate(x)
        weights = torch.softmax(weights, dim=1)
        return weights


class MOE(nn.Module):
    def __init__(self, input_dim, num_experts=20):
        super().__init__()
        self.experts = nn.ModuleList([Expert(input_dim) for _ in range(num_experts)])
        self.gating = Gating(input_dim, num_experts)

    def forward(self, x: torch.Tensor):
        # x: torch.Size([bs, 20, 320, 64, 64])
        global_latent = x[:, 0]
        num_mask = x.shape[1]
        weights = self.gating(global_latent)  # bs 20 64 64

        experts = torch.stack(
            [self.experts[i](x[:, i]) for i in range(num_mask)], dim=1
        )  # bs 20 320 64 64

        weights = weights.unsqueeze(2)  # bs 20 1 64 64
        output = torch.sum(weights * experts, dim=1) # bs 320 64 64
        return output


if __name__ == "__main__":
    model = MOE(320, 20)
    x = torch.randn(2, 20, 320, 64, 64)
    output = model(x)
    print(output.shape)  # Expected output shape: [2, 320, 64, 64]
