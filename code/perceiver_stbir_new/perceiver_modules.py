import torch
from torch import nn
import torch.nn.functional as F


class PerceiverAttentionBlock(nn.Module):
    def __init__(self, d_model, heads, dropout):
        super(PerceiverAttentionBlock, self).__init__()
        assert d_model % heads == 0, "Embedding dimension must be 0 modulo number of heads."
        self.layer_norm_x = nn.LayerNorm([d_model])
        self.layer_norm_1 = nn.LayerNorm([d_model])
        self.to_q = nn.Linear(d_model, d_model)
        self.to_kv = nn.Linear(d_model, d_model*2)
        self.attention = nn.MultiheadAttention(
            d_model,
            heads,
            dropout=0.0,
            bias=True,
            add_bias_kv=True,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(d_model, d_model)
        self.layer_norm_2 = nn.LayerNorm([d_model])
        self.linear2 = nn.Linear(d_model, d_model)
        self.linear3 = nn.Linear(d_model, d_model)

    def forward(self, x, z_input):
        x = self.layer_norm_x(x)
        z = self.layer_norm_1(z_input)
        q = self.to_q(z)
        k, v = self.to_kv(x).chunk(2, dim=-1)
        
        z, _ = self.attention(q, k, v)

        z = self.dropout(z)
        z = self.linear1(z)

        # MLP block
        z = self.layer_norm_2(z)
        z = self.linear2(z)
        z = F.gelu(z)
        z = self.dropout(z)
        z = self.linear3(z)

        return z + z_input


class PerceiverBlock(nn.Module):
    def __init__(self, d_model, latent_blocks, dropout, heads):
        super(PerceiverBlock, self).__init__()

        self.cross_attention = PerceiverAttentionBlock(
            d_model, heads=1, dropout=dropout)
        self.latent_self_attentions = nn.ModuleList([
            PerceiverAttentionBlock(d_model, heads=heads, dropout=dropout) for _ in range(latent_blocks)
        ])

    def forward(self, x, z):
        """Has a single cross-attention and a number of self latent_self_attentions

        Args:
            x (torch.Tensor): Query vector
            z (torch.Tensor): Context Vector

        Returns:
            _type_: _description_
        """
        z = self.cross_attention(x, z)
        for self_attention in self.latent_self_attentions:
            z = self_attention(z, z)
        return z


class PerceiverBlockRepeater(nn.Module):
    def __init__(self, module, repeats=1):
        super(PerceiverBlockRepeater, self).__init__()

        self.repeats = repeats
        self.module = module

    def forward(self, x, z):
        for _ in range(self.repeats):
            z = self.module(x, z)
        return z