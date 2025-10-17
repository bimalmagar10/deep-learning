from torchtext.vocab import build_vocab_from_iterator
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader as TorchDataLoader, Dataset
import random

class TranslationDataset(Dataset):
    """Custom Dataset for translation pairs"""
    def __init__(self, examples, src_vocab, trg_vocab):
        self.examples = examples
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        # Convert tokens to indices
        src_indices = [self.src_vocab[token] for token in example['src']]
        trg_indices = [self.trg_vocab[token] for token in example['trg']]
        return torch.tensor(src_indices), torch.tensor(trg_indices)

def collate_fn(batch, pad_idx_src, pad_idx_trg):
    """Collate function to pad sequences in a batch"""
    src_batch, trg_batch = [], []
    for src, trg in batch:
        src_batch.append(src)
        trg_batch.append(trg)
    
    # Pad sequences
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx_src)
    trg_batch = pad_sequence(trg_batch, batch_first=True, padding_value=pad_idx_trg)
    
    return src_batch, trg_batch

class DataLoader:
    """
    DataLoader for transformer with batching support
    
    :param init_token: start of sentence token
    :param end_token: end of sentence token
    """
    def __init__(self, init_token="<sos>", end_token="<eos>", tokenize_source=None, tokenize_target=None):
        print("Initializing the data loader........")
        self.init_token = init_token
        self.end_token = end_token
        self.tokenize_source = tokenize_source
        self.tokenize_target = tokenize_target
    
    def make_dataset(self, source, target):
        """
        Create dataset from source and target sentences
        
        Returns:
            train_data, valid_data, test_data: lists of examples
        """
        examples = []
        
        for src, trg in zip(source, target):
            examples.append({
                "src": [self.init_token] + self.tokenize_source(src) + [self.end_token],
                "trg": [self.init_token] + self.tokenize_target(trg) + [self.end_token]
            })
        
        train_size = int(0.8 * len(examples))
        valid_size = int(0.9 * len(examples))
        
        train_data = examples[:train_size]
        valid_data = examples[train_size:valid_size]
        test_data = examples[valid_size:]
        
        return train_data, valid_data, test_data
        
    def build_vocab(self, train_data, valid_data, test_data, min_freq=1):
        """
        Build vocabularies from datasets
        """
        def yield_tokens(data_iter, key):
            for data in data_iter:
                yield data[key]
        
        print("Building vocabularies...")
        
        vocab_src = build_vocab_from_iterator(
            yield_tokens(train_data + valid_data + test_data, "src"),
            min_freq=min_freq,
            specials=["<unk>", "<pad>", "<sos>", "<eos>"]
        )
        vocab_src.set_default_index(vocab_src["<unk>"])
        
        vocab_trg = build_vocab_from_iterator(
            yield_tokens(train_data + valid_data + test_data, "trg"),
            min_freq=min_freq,
            specials=["<unk>", "<pad>", "<sos>", "<eos>"]
        )
        vocab_trg.set_default_index(vocab_trg["<unk>"])
        
        print(f"Source vocabulary size: {len(vocab_src)}")
        print(f"Target vocabulary size: {len(vocab_trg)}")
        
        return vocab_src, vocab_trg
    
    def create_iterators(self, train_data, valid_data, test_data, vocab_src, vocab_trg, batch_size, device):
        train_dataset = TranslationDataset(train_data, vocab_src, vocab_trg)
        valid_dataset = TranslationDataset(valid_data, vocab_src, vocab_trg)
        test_dataset = TranslationDataset(test_data, vocab_src, vocab_trg)
        
        pad_idx_src = vocab_src['<pad>']
        pad_idx_trg = vocab_trg['<pad>']
        
        train_iterator = TorchDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, pad_idx_src, pad_idx_trg)
        )
        
        valid_iterator = TorchDataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, pad_idx_src, pad_idx_trg)
        )
        
        test_iterator = TorchDataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, pad_idx_src, pad_idx_trg)
        )
        
        return train_iterator, valid_iterator, test_iterator


