import re
import html
import contractions
import requests
from torchtext.data.utils import get_tokenizer
from pyvi.ViTokenizer import tokenize

class DataLoader:
    def __init__(self, url_en, url_vi):
        # function to preprocessing
        self.__tokenizer_en = get_tokenizer('spacy', language='en_core_web_sm')
        self.__tokenizer_vi = lambda text: list(map(lambda word: re.sub('_', ' ', word), tokenize(text).split()))

        self.__check_dict = { # bổ xung
            ' \'s': '\'s',
            '& lt ;': '<',
            '& gt ;': '>',
            "<[^<]+>":'',
            ' +': ' ',
        }

        #last run
        self.__en_data = self.__load_data(url_en, 'en')
        self.__vi_data = self.__load_data(url_vi, 'vi')

    def __text_preprocessing(self, text: str, language: str = 'en'):
        text = html.unescape(text)
        for pattern, repl in self.__check_dict.items():
            text = re.sub(pattern, repl, text)

        if language == 'en':
            text = contractions.fix(text)
            return self.__tokenizer_en(text)

        return self.__tokenizer_vi(text)

    def __load_data(self, url, language: str):
        return [self.__text_preprocessing(line, language) for line in requests.get(url).text.splitlines()]

    @property
    def vi(self):
        return self.__vi_data

    @property
    def en(self):
        return self.__en_data
