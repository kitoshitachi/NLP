from typing import Iterator, List, Optional
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from underthesea import word_tokenize

class Language:
    def __init__(self, name: str = 'en'):
        if name == 'en':
            self.__name = 'en'
            self.__tokenizer = get_tokenizer('spacy', language ='en_core_web_sm') 
        else:
            self.__name = 'vi'
            self.__tokenizer = word_tokenize

        self.__vocab = None

    def __yield_tokens(self, data):
        for line in data:
            for sentence in line:
                yield self.__tokenizer(sentence) 

    def make_vocab(self, train_iter: Iterator, min_freq:int = 1,specials: Optional[List[str]] = None, default_idx:int = 0):
        self.__vocab = build_vocab_from_iterator(self.__yield_tokens(train_iter), min_freq, specials)
        self.__vocab.set_default_index(default_idx)

    @property
    def name(self):
        return self.__name

    @property
    def vocab(self):
        return self.__vocab
    
    @property
    def tokenizer(self):
        return self.__tokenizer