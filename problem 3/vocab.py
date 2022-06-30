from typing import Iterator, List, Optional
from torchtext.vocab import build_vocab_from_iterator

class Language:
    def __init__(self, train_iter: Iterator, min_freq:int = 1):
        self.specials = ["<unk>", "<pad>", "<sos>", "<eos>"]
        self.__make_vocab(train_iter,min_freq)
    def __yield_tokens(self, data):
        for line in data:
            yield line  

    def __make_vocab(self, train_iter: Iterator, min_freq:int = 1):
        self.__vocab = build_vocab_from_iterator(self.__yield_tokens(train_iter), min_freq, self.specials)
        self.__vocab.set_default_index(0)

    @property
    def vocab(self):
        return self.__vocab
    
    def lookup_indices(self, token_list: List[str]):
        return [2,*self.vocab.lookup_indices(token_list),3]