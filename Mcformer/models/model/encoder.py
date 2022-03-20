"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from models.blocks.encoder_layer import EncoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding


class Encoder(nn.Module):

    def __init__(self, pad_unk, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()

        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        pad_unk=pad_unk,
                                        drop_prob=drop_prob,
                                        device=device)

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x, s_mask):
        x = self.emb(x)
        # print('--------------------embedding x------------------')
        #print(x.shape) #[32, 128, 512]
        # print('--------------------s_mask------------------')
        # print(s_mask.shape)  #([32, 1, 2, 128, 128])
        #
        s_mask= s_mask[:,:,0,:,:]  # [32, 1, 128, 128]  为什么不是1   在这里 [:,:,0,:,:] ==[:,:,1,:,:]

        for layer in self.layers:
            x = layer(x, s_mask)

        return x
