"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import pylab as p
import torch
from torch import nn

from models.model.decoder import Decoder
from models.model.encoder import Encoder


class Transformer(nn.Module):

    def __init__(self,  src_pad_idx,trg_pad_idx,trg_cat_sos_idx,trg_type_sos_idx,pad_unk, d_model, n_head, max_len,ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        self.src_pad_idx = src_pad_idx  # src_pad_idx是什么？pad ==1
        self.trg_pad_idx = trg_pad_idx
        self.trg_cat_sos_idx = trg_cat_sos_idx
        self.trg_type_sos_idx = trg_type_sos_idx
        self.device = device
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               pad_unk=pad_unk,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               pad_unk=pad_unk,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)
    # src:encode   trg:decode
    def forward(self, src, trg,last_word):
        # src为[cat_seq,type_seq]
        #
        #
        # (batch,2,seq_len)
        src_mask = self.make_pad_mask(src, src)

        src_trg_mask = self.make_pad_mask(trg, src)

        trg_mask = self.make_pad_mask(trg, trg) * \
                   self.make_no_peak_mask(trg, trg)

        # * \ 点乘
        # src src_mask
        # encoder x_train
        # enc_src = self.encoder(src, src_mask)
        # output = self.decoder(trg, enc_src, trg_mask, src_trg_mask)
        #

        #src_mask.shape  [32, 2, 128]
        #src_mask.shape  [32, 1, 2, 128]
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_trg_mask,last_word)
        return output

    def make_pad_mask(self, q, k):
        # if mask is not None:
        #     scores = scores.masked_fill(mask == 0, -1e9)

        len_q, len_k = q.size(2), k.size(2)
        #len_q, len_k = q.size(1), k.size(1)
        # 原来是 batch_size x len_k
        # batch_size x 1 x 1 x 2 x len_k

        # batch_size x 1 x 1 x len_k
        k = k.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(3)  #k为tensor，相等为 False ,不等为True
        # batch_size x 1 x len_q x len_k

        k = k.repeat(1, 1, 1, len_q, 1)
        # k = k.repeat(1, 1, len_q, 1)  #padd 位置  对 axis=2 复制len_q个


        # batch_size x 1 x len_q x 1
        q = q.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(4)
        # batch_size x 1 x len_q x len_k
        q = q.repeat(1, 1, 1, 1, len_k)
        #q = q.repeat(1, 1, 1, len_k)  #四维tensor,对列进行复制
        mask = k & q
        # print('--------------mask-------------')
        # print(k.shape)
        # print(q.shape)
        # print(mask.shape)
        # print(mask)
        #batch_size x 1 x 1 x 2 x len_k
        #原来batch_size x 1 x len_q x len_k
        #现在 batch_size x 1 x 2 x len_q x len_k
        # mask = mask[:,:,:,0,:]
        # 原码这个mask 为4 dim
        return mask

    def make_no_peak_mask(self, q, k):
        len_q, len_k = q.size(2), k.size(2)
        # len_q, len_k = q.size(1), k.size(1)

        # len_q x len_k
        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device)
        # 但是这个mask 为2 dim
        return mask
