from typing import Iterator
import torch
from torchtext.vocab import build_vocab_from_iterator
from typing import List
class Language:
    def __init__(self, train_iter: Iterator, min_freq:int = 1):
        Language.specials = ["<unk>", "<pad>", "<sos>", "<eos>"]
        self.__make_vocab(train_iter,min_freq)
    def __yield_tokens(self, data):
        for line in data:
            yield line  

    def __make_vocab(self, train_iter: Iterator, min_freq:int = 5):
        self.__vocab = build_vocab_from_iterator(self.__yield_tokens(train_iter), min_freq, self.specials)
        self.__vocab.set_default_index(0)
    
    @property
    def vocab(self):
        return self.__vocab
    
    def sentence_to_vector(self, sent:List[str]):
        return torch.tensor(self.__vocab.lookup_indices(sent),dtype = torch.int64)
    
    def vector_to_sentence(self, vector:List[int]):
        return self.__vocab.lookup_tokens(vector)