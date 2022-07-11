import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from perceiver_modules import PerceiverBlock, PerceiverBlockRepeater, PerceiverAttentionBlock

class VGG_Network(nn.Module):
    def __init__(self):
        super(VGG_Network, self).__init__()
        self.backbone = torchvision.models.vgg16(pretrained=True).features
        self.pool_method =  nn.AdaptiveMaxPool2d(1)

    def forward(self, input, bb_box = None):
        x = self.backbone(input)
        x = self.pool_method(x).view(-1, 512)
        return F.normalize(x)

class Txt_Encoder(nn.Module):
    def __init__(self, vocab_size, word_dim=512, output_dim=512, num_layers=1):
        super(Txt_Encoder, self).__init__()
        print 
        self.emb_layer = nn.Embedding(vocab_size, word_dim)
        self.gru_layer = nn.GRU(word_dim, 512,
            num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc_layer = nn.Linear(512*2*num_layers, output_dim)

    def forward(self, x, length):
        embedded = self.emb_layer(x) # B x max_len x word_dim
        packed = pack_padded_sequence(embedded, length.cpu(),
            batch_first=True, enforce_sorted=False)
        # packed = embedded
        # last_hidden shape: 2*2 x B x 512
        _, last_hidden = self.gru_layer(packed)
        last_hidden = torch.cat([feat for feat in last_hidden], dim=1)
        output = self.fc_layer(last_hidden)
        return output


class SetAttention(nn.Module):
    def __init__(self, input_dim, output_dim, mode='additive', num_latents=64, d_model=512, num_heads=8, num_latent_blocks=6, dropout=0.1, num_repeats=8):
        super(SetAttention, self).__init__()
        self.mode = mode
        print ('Combine Network Strategy used: ', self.mode)
        
        if self.mode == 'concat':
            self.layer = nn.Linear(input_dim*2, output_dim)
        self.init_latent = nn.Parameter(torch.rand((num_latents, d_model)))
        self.embedding = nn.Conv1d(1, d_model, 1)
        self.block1 = PerceiverBlockRepeater(
            PerceiverBlock(d_model, latent_blocks=num_latent_blocks,
                           heads=num_heads, dropout=dropout),
            repeats=1
        )  # 1
        self.block2 = PerceiverBlockRepeater(
            PerceiverBlock(d_model, latent_blocks=num_latent_blocks,
                           heads=num_heads, dropout=dropout),
            repeats=max(num_repeats - 1, 0)
        )  # 2-8

    def forward(self, input1, input2):
        if self.mode == 'concat':
            x = self.layer(torch.cat([input1, input2], dim=1))
        elif self.mode == 'additive':
            x = (input1 + input2)/2.0
            # x.shape = (batch_size, query_vector)
        else:
            raise ValueError('incorrect option')
        
        # Add cross-attention and latent transformer blocks
        x = x.unsqueeze(1)
        # x.shape = (batch_size, channels, query_vector) ; channels = 1 ; in our case
        x = self.embedding(x)
        # x.shape = (batch_size, d_model, query_vector)
        x = x.permute(2, 0, 1)
        # x.shape (query_vector, batch_size, d_model)
        
        # Transform our Z (latent)
        # z.shape = (latents, d_model)
        z = self.init_latent.unsqueeze(1)
        # z.shape = (latents, 1, d_model)
        z = z.expand(-1, x.shape[1], -1)
        # z.shape = (latents, batch_size, d_model)

        z = self.block1(x, z)
        z = self.block2(x, z)
        
        # Then average every latent
        z = z.mean(dim=0)

        return z

