import torch
from torch import nn
import torch.nn.functional as F


class PerceiverAttentionBlock(nn.Module):
    def __init__(self, d_model, heads, dropout):
        super(PerceiverAttentionBlock, self).__init__()
        assert d_model % heads == 0, "Embedding dimension must be 0 modulo number of heads."
        
        self.layer_norm_x = nn.LayerNorm([d_model])
        self.layer_norm_1 = nn.LayerNorm([d_model])
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
        z, _ = self.attention(z, x, x)

        z = self.dropout(z)
        z = self.linear1(z)

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


class Perceiver(nn.Module):

    def forward(self, x):
        batch_size = x.shape[0]
        # Transform our X (input)
        # x.shape = (batch_size, channels, width, height)

        x = x.permute(2, 0, 1)
        # x.shape (pixels, batch_size, d_model)

        # Transform our Z (latent)
        # z.shape = (latents, d_model)
        z = self.init_latent.unsqueeze(1)
        # z.shape = (latents, 1, d_model)
        z = z.expand(-1, x.shape[1], -1)
        # z.shape = (latents, batch_size, d_model)

        z = self.block1(x, z)
        z = self.block2(x, z)

        return z


class PerceiverLogits(nn.Module):
    def __init__(self, input_channels, input_shape, output_features, fourier_bands=4, latents=64, d_model=32, heads=8, latent_blocks=6, dropout=0.1, layers=8):
        super(PerceiverLogits, self).__init__()

        self.perceiver = Perceiver(
            input_channels=input_channels,
            input_shape=input_shape,
            fourier_bands=fourier_bands,
            latents=latents,
            d_model=d_model,
            heads=heads,
            latent_blocks=latent_blocks,
            dropout=dropout,
            layers=layers
        )

        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, output_features)

    def forward(self, x):
        # Run the Perceiver
        z = self.perceiver(x)

        # Let data data inside of each latent
        z = self.linear1(z)
        # Then average every latent
        z = z.mean(dim=0)
        # Then extract logits
        z = self.linear2(z)

        return F.log_softmax(z, dim=-1)