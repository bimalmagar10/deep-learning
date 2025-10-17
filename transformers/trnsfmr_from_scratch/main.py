from utils.data_loader import DataLoader
from utils.tokenize import Tokenizer
from data import en, es
from models.embedding import Embeddings
import torch
import torch.nn as nn
import sys, platform

def main():
    print("======= System Information =======")
    print(f"sys.platform: {sys.platform} and platform_machine: {platform.machine()}")
    print("This is the implementation of transformer from scratch in PyTorch!!")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Initialize tokenizer and data loader
    tokenizer = Tokenizer()
    data_loader = DataLoader(
        tokenize_source=tokenizer.tokenize_source,
        tokenize_target=tokenizer.tokenize_target
    )
    
    # Create datasets
    train_data, valid_data, test_data = data_loader.make_dataset(es, en)
    print(f"Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")
    
    # Build vocabularies
    vocab_src, vocab_trg = data_loader.build_vocab(train_data, valid_data, test_data)
    
    # Create iterators with batch_size
    batch_size = 8
    train_iterator, valid_iterator, test_iterator = data_loader.create_iterators(
        train_data, valid_data, test_data,
        vocab_src, vocab_trg,
        batch_size=batch_size,
        device=device
    )
    
    print(f"\nBatch size: {batch_size}")
    
    # Get one batch to demonstrate shape
    src_batch, trg_batch = next(iter(train_iterator))
    print(f"Batch src shape: {src_batch.shape}  <- [batch_size, seq_length]")
    print(f"Batch trg shape: {trg_batch.shape}  <- [batch_size, seq_length]")
    
    # Embedding demo: [batch_size, seq_length] -> [batch_size, seq_length, d_model]
    d_model = 512
    embed = nn.Embedding(len(vocab_src), d_model)
    embedded = embed(src_batch)
    print(f"\nAfter embedding: {embedded.shape}  <- [batch_size, seq_length, d_model]")
    print("Ready for attention mechanism!")

if __name__ == "__main__":
    main()
