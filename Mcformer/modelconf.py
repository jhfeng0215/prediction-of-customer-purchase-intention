'''配置padding unknown'''
from conf import *
import torch
# cat_pad_unk = [[0,8228,8229],  [0,217,218]]
# item_pad_unk =[[0,1704946,1704947],[0,217,218]]
# type_pad_unk =[[0,5,6], [0,217,218]]

cat_pad_unk = [[0,61,62],  [0,5089,5090]]
item_pad_unk =[[0,58973,58974],[0,5089,5090]]
type_pad_unk =[[0,5,6], [0,5089,5090]]

# 这一步之后 需要构建一个整个的dataloader
src_pad_idx = 0  # [batchsize ,len_k]  Field.vocab.stoi['<pad>']  定位pad的位置

trg_pad_idx = 0  # [batchsize ,len_k]  定位pad的位置

# 每个点击行为的嵌入维度
enc_voc_size = vac_size
dec_voc_size = vac_size

cat_pad_idx=cat_pad_unk[0][0]
cat_time_sos_idx=cat_pad_unk[0][2]
cat_type_sos_idx=cat_pad_unk[1][2]

item_pad_idx=item_pad_unk[0][0]
item_time_sos_idx=item_pad_unk[0][2]
item_type_sos_idx=item_pad_unk[1][2]

type_pad_idx=type_pad_unk[0][0]
type_time_sos_idx=type_pad_unk[0][2]
type_type_sos_idx=type_pad_unk[1][2]