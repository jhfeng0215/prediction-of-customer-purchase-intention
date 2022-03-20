import torch
from torch import nn
from data import *
from conf import *
from modelconf import *
import math
from models.model.transformer import Transformer


class Fcnn(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, cat_pad_idx, cat_time_sos_idx, cat_type_sos_idx,
                 item_pad_idx, item_time_sos_idx, item_type_sos_idx, type_pad_idx, type_time_sos_idx, type_type_sos_idx,
                 max_len, d_model,d_model_1, ffn_hidden, n_heads, n_layers, drop_prob, device):
        super().__init__()
        # self.src_pad_idx = src_pad_idx  # src_pad_idx是什么？pad ==1
        # self.trg_pad_idx = trg_pad_idx
        # self.trg_cat_sos_idx = trg_cat_sos_idx
        # self.trg_type_sos_idx = trg_type_sos_idx
        self.device = device
        self.transformer_cat = Transformer(src_pad_idx=src_pad_idx,
                                           trg_pad_idx=cat_pad_idx,
                                           trg_cat_sos_idx=cat_time_sos_idx,
                                           trg_type_sos_idx=cat_type_sos_idx,
                                           d_model=d_model,
                                           pad_unk=cat_pad_unk,
                                           max_len=max_len,
                                           ffn_hidden=ffn_hidden,
                                           n_head=n_heads,
                                           n_layers=n_layers,
                                           drop_prob=drop_prob,
                                           device=device).to(device)

        self.transformer_item = Transformer(src_pad_idx=src_pad_idx,
                                            trg_pad_idx=item_pad_idx,
                                            trg_cat_sos_idx=item_time_sos_idx,
                                            trg_type_sos_idx=item_type_sos_idx,
                                            d_model=d_model,
                                            pad_unk=item_pad_unk,
                                            max_len=max_len,
                                            ffn_hidden=ffn_hidden,
                                            n_head=n_heads,
                                            n_layers=n_layers,
                                            drop_prob=drop_prob,
                                            device=device).to(device)

        self.transformer_type = Transformer(src_pad_idx=src_pad_idx,
                                            trg_pad_idx=type_pad_idx,
                                            trg_cat_sos_idx=type_time_sos_idx,
                                            trg_type_sos_idx=type_type_sos_idx,
                                            d_model=d_model,
                                            pad_unk=type_pad_unk,
                                            max_len=max_len,
                                            ffn_hidden=ffn_hidden,
                                            n_head=n_heads,
                                            n_layers=n_layers,
                                            drop_prob=drop_prob,
                                            device=device).to(device)


        self.softmax = nn.Softmax()
        self.linear = nn.Linear(max_len*5, max_len,bias=True)
        self.linear1 = nn.Linear(d_model, 2,bias=True)
        self.linear2 = nn.Linear(d_model_1, 1,bias=True)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.lkrelu = nn.LeakyReLU()
        self.gelu = nn.GELU()
        self.sft  = nn.Softplus()
        self.tanh = nn.Tanh()

    import torch
    import torch.nn.functional as F
    def attention(self, q, k, v):
        # 获得注意力权重

        alpha_mat = torch.matmul(q, k.transpose(1, 2))  # [16,28,28]
        # 归一化注意力权重,alpha把第一求和，softmax把第二位归一化
        # print(alpha_mat.shape)
        alpha = self.softmax(alpha_mat / math.sqrt(d_model))  # [16,1,28]
        # 进行加权和
        x = torch.matmul(alpha, v)  # [16,300]   128 2

        return x

    def forward(self, X_item_train_en, X_item_train_de, X_cat_train_en, X_cat_train_de, X_type_train_en,
                X_type_train_de, X_train_wlen):
        # model(src, trg, last_word)
        output_cat, last_word = self.transformer_cat(X_cat_train_en, X_cat_train_de,
                                                     X_train_wlen)  # [32, 2]   #[32, 128, 2]
        output_item, _ = self.transformer_item(X_item_train_en, X_item_train_de, X_train_wlen)
        output_type, _ = self.transformer_type(X_type_train_en, X_type_train_de, X_train_wlen)

        type_item = self.attention(output_type, output_item, output_item)
        type_cat = self.attention(output_type, output_cat, output_cat)


        type_item = output_cat.mul(type_cat)
        type_cat = output_item.mul(type_item)


        output = type_cat + type_item + output_type

        output = self.linear1(output)
        # output = self.gelu(output)
        last_word = last_word.data.cpu().numpy()
        a = []
        for i in range(0, len(output)):
            # a.append(output[i, last_word[i], :])
            # a.append(torch.mean(output[i, :last_word[i], :],dim=0))
            
            a.append(torch.mean(output[i, :, :], dim=0))
        output = torch.stack(a)
        output = self.softmax(output)
        
        return torch.log(output)
        # return output
