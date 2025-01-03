from torch import nn as nn
import torch

class Proj(nn.Module):
    def __init__(self,embed_dim:int,hidden_dim:int):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(embed_dim,hidden_dim),
            nn.Linear(hidden_dim,embed_dim),
            nn.LayerNorm(embed_dim),
        )
    
    def forward(self,x:torch.Tensor):
        return self.feat.forward(x)