from torchtext.vocab import build_vocab_from_iterator
class DataLoader:
    """
    :param init_token: start of sentence token
    :param end_token: end of sentence token
    """
    def __init__(self,init_token="<sos>",end_token="<eos>",tokenize_source=None,tokenize_target=None):
        print("Initializing the data loader........")
        self.init_token = init_token
        self.end_token = end_token
        self.tokenize_source = tokenize_source
        self.tokenize_target = tokenize_target
    
    def make_dataset(self,source,target):
        examples = []
        for src,trg in zip(source,target):
            examples.append({
                "src": [self.init_token] + self.tokenize_source(src) + [self.end_token],
                "trg": [self.init_token] + self.tokenize_target(trg) + [self.end_token]
            })
        train_size = int(0.8 * len(examples))
        valid_size = int(0.9 * len(examples))
        train_set = examples[0:train_size]
        valid_set = examples[train_size:valid_size]
        test_set = examples[valid_size:]
        return train_set,valid_set,test_set
        
    def build_vocab(self,train_data,valid_data,test_data):
        def yield_tokens(data_iter,key):
            for data in data_iter:
                yield data[key]

        # 1. Building Spanish Vocbulary
        print("Building Spanish Vocabulary.....")
        vocab_src = build_vocab_from_iterator(
            yield_tokens(train_data + valid_data + test_data,"src"),
            min_freq = 2,
            specials = ["<unk>","<pad>","<sos>","<eos>"]
        )
        print("Done Building Spanish Vocabulary!")
        # 2. Building English Vocabulary
        print("Buidling English Vocabulary.........")
        vocab_trg = build_vocab_from_iterator(
            yield_tokens(train_data + valid_data + test_data,"trg"),
            min_freq = 2,
            specials = ["<unk>","<pad>","<sos>","<eos>"]
        )
        print("Done building English Vocabulary!")
        return vocab_src,vocab_trg


