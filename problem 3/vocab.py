from typing import Iterator, Optional
from torchtext.data.utils import get_tokenizer
from pyvi.ViTokenizer import tokenize
import re
from torchtext.vocab import build_vocab_from_iterator

tokenizer = {
    'en': get_tokenizer('spacy', language ='en_core_web_sm'),
    'vi': lambda text : list(map(lambda word: re.sub('_', ' ',word),tokenize(text).split()))
}

def yield_tokens(train_data, language='en'):
    for line in train_data:
        yield tokenizer[language](line)  

def make_vocab(train_iter: Iterator, language:str = 'en', min_freq:int = 1,specials: Optional[List[str]] = None, default_idx:int = 0):
    vocab = build_vocab_from_iterator(yield_tokens(train_iter, language), min_freq, specials)
    vocab.set_default_index(default_idx)
    return vocab