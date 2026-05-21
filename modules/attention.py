import torch
import torch.nn as nn
import torch.nn.functional as F
from bitnet import BitLinear

class Attention(nn.Module):
    def __init__(self, in_features, out_features, use_bitnet=False):
        super().__init__()
        self.q = nn.Linear(in_features, out_features) if not use_bitnet else BitLinear(in_features, out_features)
        self.k = nn.Linear(in_features, out_features) if not use_bitnet else BitLinear(in_features, out_features)
        self.v = nn.Linear(in_features, out_features) if not use_bitnet else BitLinear(in_features, out_features)
        self.d_k = out_features

        # self.softmax = nn.Softmax(out_features)
    
    def forward(self, x: torch.Tensor, attn_mask=None):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, attn_mask=attn_mask)


class MultiHeadAttention(nn.Module):
    def __init__(self, in_features, out_features, heads=4, use_bitnet=False):
        super().__init__()

        self.attn = nn.ModuleList([Attention(in_features, out_features // heads, use_bitnet) for _ in range(heads)])
    
    def forward(self, x):
        attn = [layer(x) for layer in self.attn]

        return torch.cat(attn, dim=-1)

class SpatialAttention(nn.Module):
    def __init__(self, dim=128, heads=4, use_bitnet=True):
        super().__init__()
        self.attn = MultiHeadAttention(dim, dim, heads=heads, use_bitnet=True)
    
    def forward(self, x):  # (B, C, H, W)
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        return (tokens + self.attn(tokens)).transpose(1, 2).reshape(B, C, H, W)