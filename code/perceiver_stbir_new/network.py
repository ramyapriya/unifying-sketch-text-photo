import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from perceiver_modules import Attention, FeedForward, PreNorm, cache_fn
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
        self.emb_layer = nn.Embedding(vocab_size, word_dim)
        self.gru_layer = nn.GRU(word_dim, 512,
            num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc_layer = nn.Linear(512*2*num_layers, output_dim)

    def forward(self, x, length):
        embedded = self.emb_layer(x) # B x max_len x word_dim
        try:
            length = length.cpu()
        except:
            pass
        packed = pack_padded_sequence(embedded, length,
            batch_first=True, enforce_sorted=False)
        # packed = embedded
        # last_hidden shape: 2*2 x B x 512
        _, last_hidden = self.gru_layer(packed)
        last_hidden = torch.cat([feat for feat in last_hidden], dim=1)
        output = self.fc_layer(last_hidden)
        return output


class SetAttention(nn.Module):
    def __init__(self, input_dim, output_dim, mode='additive', **kwargs):
        super(SetAttention, self).__init__()
        self.mode = mode
        for attr, value in kwargs.items():
            setattr(self, attr, value)
            
        # if self.mode == 'concat':
        #     self.layer = nn.Linear(input_dim*2, output_dim)
            
        print ('Combine Network Strategy used: ', self.mode)
        self.init_latent = nn.Parameter(torch.rand(self.num_latents, self.latent_dim))
        get_cross_attn_first = lambda: PreNorm(self.latent_dim, Attention(self.latent_dim, input_dim, heads = self.cross_heads, dim_head = self.cross_dim_head, dropout = self.attn_dropout), context_dim = input_dim)
        get_cross_ff_first = lambda: PreNorm(self.latent_dim, FeedForward(self.latent_dim, dropout = self.ff_dropout))
        get_latent_attn_first = lambda: PreNorm(self.latent_dim, Attention(self.latent_dim, heads = self.latent_heads, dim_head = self.latent_dim_head, dropout = self.attn_dropout))
        get_latent_ff_first = lambda: PreNorm(self.latent_dim, FeedForward(self.latent_dim, dropout = self.ff_dropout))
        
        get_cross_attn = lambda: PreNorm(self.latent_dim, Attention(self.latent_dim, input_dim, heads = self.cross_heads, dim_head = self.cross_dim_head, dropout = self.attn_dropout), context_dim = input_dim)
        get_cross_ff = lambda: PreNorm(self.latent_dim, FeedForward(self.latent_dim, dropout = self.ff_dropout))
        get_latent_attn = lambda: PreNorm(self.latent_dim, Attention(self.latent_dim, heads = self.latent_heads, dim_head = self.latent_dim_head, dropout = self.attn_dropout))
        get_latent_ff = lambda: PreNorm(self.latent_dim, FeedForward(self.latent_dim, dropout = self.ff_dropout))

        get_cross_attn_first, get_cross_ff_first, get_latent_attn_first, get_latent_ff_first = map(cache_fn, (get_cross_attn_first, get_cross_ff_first, get_latent_attn_first, get_latent_ff_first))
        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        
        # Layer 1
        should_cache = False
        cache_args = {'_cache': should_cache}
        
        self_attns = nn.ModuleList([])
        for block_ind in range(self.self_per_cross_attn):
            self_attns.append(nn.ModuleList([
                get_latent_attn_first(**cache_args, key = block_ind),
                get_latent_ff_first(**cache_args, key = block_ind)
            ]))
        self.layers.append(nn.ModuleList([
                get_cross_attn_first(**cache_args),
                get_cross_ff_first(**cache_args),
                self_attns
            ]))
        
        # Layer 2 to depth - share weights
        for i in range(self.depth-1):
            should_cache = i > 0 and self.weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for block_ind in range(self.self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args, key = block_ind),
                    get_latent_ff(**cache_args, key = block_ind)
                ]))

            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                self_attns
            ]))

        self.embedding = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, output_dim)
        )

    def forward(self, input1, input2):
        if self.mode == 'concat':
            x = self.layer(torch.cat([input1, input2], dim=1))
            # x.shape = (batch_size, (input1 + input2))
        elif self.mode == 'additive':
            x = (input1 + input2)/2.0
            # x.shape = (batch_size, input1(or 2)) - both vectors MUST have same size
        
        x = x.unsqueeze(1)
        # x.shape = (batch_size, 1, query_vector) ; channels = 1 ; in our case
        
        # Transform our Z (latent)
        # z.shape = (latents, d_model)
        z = self.init_latent.unsqueeze(0)
        # z.shape = (1, latents, d_model)
        z = z.expand(x.shape[0],-1,-1)
        # z.shape = (batch_size, latents, d_model)
        
        # Add cross-attention and latent transformer blocks
        
        for cross_attn, cross_ff, self_attns in self.layers:
            z = cross_attn(z, context=x, mask=None) + z
            z = cross_ff(z) + z

            for self_attn, self_ff in self_attns:
                z = self_attn(z) + z
                z = self_ff(z) + z
        z = torch.mean(z, axis=1)
        z = self.embedding(z)
        return z
    
if __name__ == "__main__":
    net = SetAttention(512, 512)
    
    sk = torch.rand(10, 512)
    im = torch.rand(10, 512)
    q = net(sk, im)
    print(q.shape)
    

