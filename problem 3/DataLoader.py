import re
import html
import contractions
import requests

def text_preprocessing(text, language = 'en'):
    if language == 'en':
        return re.sub(' +', ' ', contractions.fix(html.unescape(text)))
    else:
        return re.sub(' +', ' ',html.unescape(text))

def Data_Iter(url,language = 'en'):
    for line in requests.get(url).text.splitlines():
        yield text_preprocessing(line,language)


