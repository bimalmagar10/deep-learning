from utils.data_loader import DataLoader
from utils.tokenize import Tokenizer
from data import en,es
from models.embedding import Embeddings
import torch
import torch.nn as nn
import sys,platform

def main():
    print(f"sys.platform: {sys.platform} and platform_machine: {platform.machine()}")
    print("This is the implementation of transformer from scratch in PyTorch!!")
    tokenizer = Tokenizer()
    data_loader = DataLoader(
        tokenize_source=tokenizer.tokenize_source,
        tokenize_target=tokenizer.tokenize_target
    )
    train_set,valid_set,test_set = data_loader.make_dataset(es,en)
    vocab_src,_ = data_loader.build_vocab(train_set,valid_set,test_set)
    embed = nn.Embedding(len(vocab_src),4)
    print(embed(torch.tensor(vocab_src.get_stoi()['víctimas'])))
if __name__ == "__main__":
    main()