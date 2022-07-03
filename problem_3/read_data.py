import re
import html
import contractions
import requests
from torchtext.data.utils import get_tokenizer
from pyvi.ViTokenizer import tokenize

class DataLoader:
    def __init__(self, url_en:str, url_vi:str):
        # function to preprocessing
        DataLoader.__tokenizer_en = get_tokenizer('spacy', language='en_core_web_sm')
        DataLoader.__tokenizer_vi = lambda text: list(map(lambda word: re.sub('_', ' ', word), tokenize(text).split()))

        DataLoader.__check_dict = { # bổ xung
            ' \'s': '\'s',
            '& lt ;': '<',
            '& gt ;': '>',
            "<[^<]+>":'',
            ' +': ' ',
        }

        #last run
        self.__load_data(url_en, url_vi)

    def __text_preprocessing(self, text: str, language: str = 'en'):
        text = html.unescape(text)
        for pattern, repl in DataLoader.__check_dict.items():
            text = text.lower()
            text = re.sub(pattern, repl, text)

        if language == 'en':
            text = contractions.fix(text)
            return DataLoader.__tokenizer_en(text)

        return DataLoader.__tokenizer_vi(text)

    def __load_data(self, url_en:str, url_vi:str):
        data_en = requests.get(url_en).text.strip().splitlines()
        data_vi = requests.get(url_vi).text.strip().splitlines()
        self.__en_data = []
        self.__vi_data = []
        for en,vi in zip(data_en,data_vi):
            en = ["<sos>",*self.__text_preprocessing(en, 'en'), "<eos>"] 
            vi = ["<sos>",*self.__text_preprocessing(vi, 'vi'), "<eos>"] 
            if len(en) < 33 and len(vi) < 33:
                self.__en_data.append(en)
                self.__vi_data.append(vi)

    @property
    def vi(self):
        return self.__vi_data

    @property
    def en(self):
        return self.__en_data
    
    @property
    def data(self):
        '''return en_data, vi_data'''
        return list(zip(self.__en_data,self.__vi_data))
