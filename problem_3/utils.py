import re
import html
import contractions
import requests

class Data:
	def __init__(self, url_en:str, url_vi:str):
		Data.__check_dict = { # bổ xung
		' \'s': '\'s',
		'& lt ;': '<',
		'& gt ;': '>',
		"<[^<]+>":'',
		' +': ' ',
		}

		data_en = requests.get(url_en).text.strip().splitlines()
		data_vi = requests.get(url_vi).text.strip().splitlines()
		self.__data_en = [self.__text_preprocessing(en, 'en') for en in data_en]
		self.__data_vi = [self.__text_preprocessing(vi, 'vi') for vi in data_vi]

	def __text_preprocessing(self, text: str, language: str = 'en'):
		text = html.unescape(text)
		for pattern, repl in Data.__check_dict.items():
			text = text.lower()
			text = re.sub(pattern, repl, text)

		if language == 'en':
			text = re.sub(' +', ' ', contractions.fix(text))

		return text

	@property
	def en(self):
		return self.__data_en

	@property
	def vi(self):
		return self.__data_vi

def make_batch(data, batchsize = 32):
	bb = []
	ben = []
	bvi = []
	for en, vi in data: 
		ben.append(en)
		bvi.append(vi)
		if len(ben) >= batchsize:
			bb.append((ben, bvi))
			ben = []
			bvi = []
	if len(ben) > 0:
		bb.append((ben, bvi))
	return bb


def padding_batch(b):
	maxlen = max([len(x) for x in b])
	for tkl in b:
		for i in range(maxlen - len(tkl)):
			tkl.append('<pad>')

def padding(bb):
	for ben, bvi in bb:
		padding_batch(ben)
		padding_batch(bvi)