import torch.nn as nn
import math
class Embeddings(nn.Module):
    def __init__(self,vocab_size,d_model):
        super().__init__()
        print("Initialized embeddings class!!!")
        self.d_model = d_model
        self.vocab_size =  vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)
    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)