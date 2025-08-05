import math

"""
Simple MLP Implementation
"""
import torch
import torch.nn as nn

class MLPPyTorch(nn.Module):
    """
    Same MLP but using PyTorch - much cleaner!
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPPyTorch, self).__init__()
        
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Same logic, but PyTorch handles the math
        """
        # Step 1: Input → Hidden layer
        hidden = self.layer1(x)  # Automatically does: x * W1 + b1
        
        # Step 2: Apply ReLU
        hidden = self.relu(hidden)  # ReLU activation
        
        # Step 3: Hidden → Output layer  
        output = self.layer2(hidden)  # Automatically does: hidden * W2 + b2
        
        return output
   