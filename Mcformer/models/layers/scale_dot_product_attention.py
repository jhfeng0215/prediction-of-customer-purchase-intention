"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import math

from torch import nn


class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]

        #torch.Size([32, 8, 128, 64])
        batch_size, head, length, d_tensor = k.size()  #[32, 8, 128, 64]

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose  #torch.Size([32, 8, 64, 128])
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # print(score.shape)

        # 2. apply masking (opt)
        # print(mask.shape)
        # print(mask)
        # print(score)
        # print(mask.shape)
        # print(score.shape)
        if mask is not None:
            score = score.masked_fill(mask == 0, -e)

        # print('---------score---------------')
        # print(score.shape)
        # print(score)
        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v
        # print(v)
        return v, score
