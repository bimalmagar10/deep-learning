import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    
    INPUT SHAPE:
    ------------
    q (query): [batch_size, seq_length, d_model]
    k (key):   [batch_size, seq_length, d_model]
    v (value): [batch_size, seq_length, d_model]
    
    OUTPUT SHAPE:
    -------------
    output: [batch_size, seq_length, d_model]
    
    Note: Multi-head dimension will be added in MultiHeadAttention module
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, k, v):
        """
        Args:
            q: [batch_size, seq_length, d_model]
            k: [batch_size, seq_length, d_model]
            v: [batch_size, seq_length, d_model]
            
        Returns:
            output: [batch_size, seq_length, d_model]
            attention_weights: [batch_size, seq_length, seq_length]
        """
        # Linear transformations
        query = self.w_q(q)  # [batch_size, seq_length, d_model]
        key = self.w_k(k)    # [batch_size, seq_length, d_model]
        value = self.w_v(v)  # [batch_size, seq_length, d_model]
        
        # Compute attention scores: Q @ K^T
        # [batch_size, seq_length, d_model] @ [batch_size, d_model, seq_length]
        # -> [batch_size, seq_length, seq_length]
        scores = query @ key.transpose(-2, -1)
        
        # Scale by sqrt(d_model)
        scores = scores / math.sqrt(self.d_model)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # -> [batch_size, seq_length, seq_length]
        
        # Apply attention weights to values
        # [batch_size, seq_length, seq_length] @ [batch_size, seq_length, d_model]
        output = attention_weights @ value # -> [batch_size, seq_length, d_model]
        
        return output, attention_weights
    
    def query_key_value_attention(self, q, k,v, scaled=True):
        
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        scores = query @ key.transpose(-2, -1)
        
        if scaled:
            scores = scores / math.sqrt(self.d_model)
        
        attened_scores = F.softmax(scores, dim=-1)

        output = attened_scores @ value

        return attened_scores, output


