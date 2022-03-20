import torch
from torch import nn
from data import *

from models.model.transformer import Transformer
transformer = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_cat_sos_idx=trg_cat_sos_idx,
                    trg_type_sos_idx=trg_type_sos_idx,
                    d_model=d_model,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device).to(device)
class Output_att(nn.Module):
    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.linear=nn.MultiheadAttention()


    def forward(self, X_item_train_en, X_item_train_de, X_cat_train_en, X_cat_train_de,X_type_train_en,
            X_type_train_de,X_train_wlen):
        # model(src, trg, last_word)
        output_cat = transformer(X_cat_train_en,X_cat_train_de,X_train_wlen)   #[32, 2]
        output_item = transformer(X_item_train_en,X_item_train_de,X_train_wlen)
        output_type = transformer(X_type_train_en,X_type_train_de,X_train_wlen)
        output_time = transformer()
        # 拼接 [32,4,2]
        output = torch.stack((output_cat, output_item,output_type,output_time), dim=1)
        output=output.transpose(1, 2)
        output=self.linear(output)   #[32,2]
        return output