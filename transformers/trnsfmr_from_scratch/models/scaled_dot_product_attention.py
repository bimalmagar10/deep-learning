import torch.nn as nn

# Scaled Dot Product Attention Module from transformers
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w_q = nn.Linear(d_model,d_model,bias = False) # Query weights
        self.w_k = nn.Linear(d_model,d_model,bias=False) # Key weights
        self.w_v = nn.Linear(d_model,d_model,bias=False) # Value weights

    def forward(self,q,k,v):
        # These are the query, key and value after the weights are multiplied
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        return query,key,value
