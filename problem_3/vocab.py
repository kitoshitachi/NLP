from spacy import Vocab
from torchtext.vocab import build_vocab_from_iterator
from typing import List
from torchtext.data import get_tokenizer
from utils import Data
class Language:
  def __init__(self, data: Data, min_freq:int = 1):
    specials = ["<unk>", "<pad>", "<sos>", "<eos>"]
    Language.__tokenizer = {
      'vi': lambda text: list(map(lambda word: re.sub('_', ' ', word), ViTokenizer.tokenize(text).split())),
      'en': get_tokenizer('spacy', language='en_core_web_sm')
    }
    self.__vocab_en = build_vocab_from_iterator(self.__yield_tokens(data.en,'en'), min_freq, specials)
    self.__vocab_en.set_default_index(0)

    self.__vocab_vi = build_vocab_from_iterator(self.__yield_tokens(data.vi,'vi'), min_freq, specials)
    self.__vocab_vi.set_default_index(0)
  
  def __yield_tokens(self, data:List[str] , language:str = 'en'):
    for line in data:
      yield Language.__tokenizer[language](line)

  @property
  def en(self):
    return self.__vocab_en

  @property
  def vi(self):
    return self.__vocab_vi
  
  def text_pipeline(self, data:List[str], language:str = 'en') -> List[str]:
    if language == 'en':
      return [self.__vocab_en.lookup_tokens([2,*self.__vocab_en.lookup_indices(Language.__tokenizer[language](line)),3]) for line in data]
    if language == 'vi':
      return [self.__vocab_vi.lookup_tokens([2,*self.__vocab_vi.lookup_indices(Language.__tokenizer[language](line)),3]) for line in data]