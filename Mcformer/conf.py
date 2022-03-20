"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import torch
import os



is_test=True
# GPU device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
# model parameter setting
batch_size = 128
max_len = 100  #句子长度  平均点击长度为32
d_model = 256  #embedding 长度
d_model_1 = 100    #d_model_1 = max_len
vac_size = 256
n_layers =2
n_heads = 8
ffn_hidden = 756
drop_prob = 0.25


# optimizer parameter setting
# init_lr =  0.0002
# # factor = 0.75
# factor = 0.75
# adam_eps = 1e-8
# # adam_eps = 0
# # patience = 6
# patience = 6
# # warmup = 100
# epoch = 100
# # warmup = 100
# # epoch = 5
# warmup = 100
# clip = 2
# #clip = 2
# weight_decay = 0
# inf = float('inf')




init_lr =  0.0002
# factor = 0.75
factor = 0.75
adam_eps = 1e-8
# adam_eps = 0
# patience = 6
patience = 6
# warmup = 100
epoch = 100
# warmup = 100
# epoch = 5
warmup = 100
clip = 1
#clip = 2
weight_decay = 0.5
inf = float('inf')