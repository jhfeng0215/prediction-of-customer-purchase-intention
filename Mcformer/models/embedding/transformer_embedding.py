"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
from torch import nn
import torch
import torch
import torch.nn.functional as F

from models.embedding.positional_encoding import PostionalEncoding
from models.embedding.category_embeddings import CategoryEmbedding
from models.embedding.clctype_embedding import ClicktypeEmbedding



class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    """
    # 对x进行分解
    # x= [[][]
    #     [][]
    #     [][]]
    def __init__(self, pad_unk, d_model, max_len, drop_prob, device):
        """
        class for word embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TransformerEmbedding, self).__init__()
        self.cat_emb = CategoryEmbedding(pad_unk[0][2]+1, d_model)

        self.clk_emb = ClicktypeEmbedding(pad_unk[1][2]+1, d_model)
        self.timeint_emb = ClicktypeEmbedding(282121, d_model)
        self.pos_emb = PostionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)
        self.relu = nn.ReLU()

    def forward(self, x):
        #对x切分在进行相加 mask取其中一维


        x_cat=x[:,0,:]
        x_clk=x[:,1,:]
        x_timeint=x[:,2,:]

        clk_emb = self.clk_emb(x_clk)
        cat_emb = self.cat_emb(x_cat)
        pos_emb = self.pos_emb(x_cat)
        x_timeint_emb = self.timeint_emb(x_timeint)
        
        
        # a=x_timeint.dtype
        # x_timeint=x_timeint.type(torch.float32)
        # x_timeint_emb=F.normalize(x_timeint,p=2,dim=0)
        # x_timeint_emb = x_timeint_emb.type(a)
        # x_timeint_emb=x_timeint_emb.repeat(256,1,1).permute(1,2,0)   #d_model
        
        
#         return self.drop_out(cat_emb+pos_emb+clk_emb)
        # output=self.relu(cat_emb+pos_emb+clk_emb+x_timeint.T)
        # print(cat_emb.shape)
        # print(pos_emb.shape) 
        # print(clk_emb.shape)
        # print(x_timeint_emb.shape)
        output=cat_emb+pos_emb+clk_emb+x_timeint_emb
        # output=cat_emb+pos_emb+clk_emb
        # print(cat_emb.shape)  #torch.Size([128, 60, 128])
        # print(pos_emb.shape)  #torch.Size([60, 128])
        # print(clk_emb.shape)  # torch.Size([128, 60, 128])
        # print(x_timeint_emb.shape)  #torch.Size([128, 60])
        



        # output= cat_emb+pos_emb+clk_emb+x_timeint_emb.T
        # output= cat_emb+pos_emb+clk_emb
        # return self.drop_out(self.relu(output))
        return self.drop_out(output)
#         return self.drop_out(self.relu(cat_emb+pos_emb + x_clk))
            
#         print(x_clk.shape)
#         print(cat_emb.shape)     #[128, 80, 128]
#         print(pos_emb.shape)     #[80, 128]
#         print(x_clk.shape)       #[128, 80]
        
