import spacy
class Tokenizer:
    def __init__(self):
        self.tokenize_espaneol = spacy.load("es_core_news_sm")
        self.tokenize_english = spacy.load("en_core_web_sm")
    def tokenize_source(self,text):
        return [tok.text for tok in self.tokenize_espaneol.tokenizer(text)]
    def tokenize_target(self,text):
        return [tok.text for tok in self.tokenize_english.tokenizer(text)]