from DataLoader import Data_Iter
from vocab import make_vocab

url = "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/"

train_en, train_vi = Data_Iter(url + 'train.en','en'), Data_Iter(url + 'train.vi','vi')

test_en, test_vi = Data_Iter(url + 'tst2013.en'), Data_Iter(url + 'tst2013.vi')

UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
specials = ["<unk>", "<pad>", "<sos>", "<eos>"]

vocab_en = make_vocab(train_en,'en',3,specials,UNK_IDX)
vocab_vi = make_vocab(train_vi,'vi',3,specials,UNK_IDX)

print("vocab size en:", len(vocab_en.get_itos()))
print("vocab size vi:", len(vocab_vi.get_itos()))
