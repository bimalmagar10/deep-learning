import torch.nn as nn

# Scaled Dot Product Attention Module from transformers
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super()
        self.w_q = None # Query weights
        self.w_k = None # Key weights
        self.w_v = None # Vaue weights

    def forward(self,query,key,value):
        pass
