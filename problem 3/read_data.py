import re
import html
import contractions
import requests

class DataLoader:
    def __init__(self, url_en, url_vi):
        self.__en_data = self.__load_data(url_en,'en')
        self.__vi_data = self.__load_data(url_vi,'vi')

    def __text_preprocessing(self, text:str, language:str = 'en'):
        text = html.unescape(text)
        text = re.sub(' +', ' ', text)
        if language == 'en':
            from underthesea import sent_tokenize
            text = contractions.fix(text)
        else:
            from nltk import sent_tokenize

        return [sentence for sentence in sent_tokenize(text)]

    def __load_data(self, url, language:str):
        return [self.__text_preprocessing(line,language) for line in requests.get(url).text.splitlines()]

    @property
    def vi(self):
        return self.__vi_data

    @property
    def en(self):
        return self.__en_data