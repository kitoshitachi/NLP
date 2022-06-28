from read_data import DataLoader
from vocab import Language

#read data
url = "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/"

train = DataLoader(url +'train.en',url +'train.vi')
test = DataLoader(url + 'tst2013.en',url + 'tst2013.vi')

#check data
print(train.vi[0])
print(len(train.en))

#make vocab
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
specials = ["<unk>", "<pad>", "<sos>", "<eos>"]

Vi = Language('vi')
En = Language('en')
Vi.make_vocab(train.vi,3,specials,UNK_IDX)
En.make_vocab(train.en,3,specials,UNK_IDX)

#check vocab
print("vocab size vi:", len(Vi.vocab.get_itos()))
print("vocab size en:", len(En.vocab.get_itos()))
for word in Vi.vocab.get_itos()[:5]:
    print(word,Vi.vocab[word])