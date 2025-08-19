import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Scaled Dot Product Attention Module from transformers
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.w_q = nn.Linear(d_model,d_model,bias = False) # Query weights
        self.w_k = nn.Linear(d_model,d_model,bias=False) # Key weights
        self.w_v = nn.Linear(d_model,d_model,bias=False) # Value weights

    def forward(self,q,k,v):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        
        scores = query @ key.transpose(-2,-1)
        
        # Scaling the score as in paper
        scores = scores / math.sqrt(self.d_model)
        attention_weights = F.softmax(scores, dim=-1)
        output = attention_weights @ value
        
        return output
    
    def query_key_attention(self,q,k,scaled=True):

        query = self.w_q(q)
        key = self.w_k(k)
        scores = query @ key.transpose(-2,-1)
        if scaled:
            scores = scores / math.sqrt(self.d_model)
        return scores
