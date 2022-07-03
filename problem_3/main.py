from read_data import DataLoader as read
from vocab import Language
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

import torch
MODEL_NAME = "nlp.model"
EPOCH = 10
BATCHSIZE = 128
LR = 0.0001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


#read data
url = "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/"

train = read(url +'train.en',url +'train.vi')
val = read(url + 'tst2012.en',url + 'tst2012.vi')
test = read(url + 'tst2013.en',url + 'tst2013.vi')

#make vocab
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3

Vi = Language(train.vi,3)
En = Language(train.en,3)

#data preprocess

train_en = [En.sentence_to_vector(line) for line in train.en]
train_vi = [Vi.sentence_to_vector(line) for line in train.vi]

#padding
train_en = pad_sequence(train_en,batch_first= True,padding_value=UNK_IDX)
train_vi = pad_sequence(train_vi,batch_first= True,padding_value=UNK_IDX)

#make batch
train_data = list(zip(train_en, train_vi))
train_data = list(DataLoader(train_data,batch_size=BATCHSIZE,shuffle=True))