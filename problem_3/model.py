import torch
from torch.nn.functional import cross_entropy

from vocab import Language

class LSTM(torch.nn.Module):
	def __init__(self, vocablist_x, vocabidx_x, vocablist_y, vocabidx_y):
		super(LSTM, self).__init__()

		self.encemb = torch.nn.Embedding(len(vocablist_x), 256, padding_idx = vocabidx_x['<pad>'])
		self.dropout = torch.nn.Dropout(0.5)
		self.enclstm = torch.nn.LSTM(256,516,2,dropout=0.5)
		
		self.decemb = torch.nn.Embedding(len(vocablist_x), 256, padding_idx = vocabidx_y['<pad>'])
		self.declstm = torch.nn.LSTM(256,516,2,dropout=0.5)
		self.decout = torch.nn.Linear(516, len(vocabidx_y))
  
	def forward(self,x, device:str):
		x, y = x[0], x[1]
		# print(x.size())
		# print(y.size())

		e_x = self.dropout(self.encemb(x))
		
		outenc,(hidden,cell) = self.enclstm(e_x)

		n_y=y.shape[0]
		loss = torch.tensor(0.,dtype=torch.float32).to(device)
		for i in range(n_y-1):
			input = y[i]
			input = input.unsqueeze(0)
			input = self.dropout(self.decemb(input))
			outdec, (hidden,cell) = self.declstm(input,(hidden,cell))
			output = self.decout(outdec.squeeze(0))
			input = y[i+1]
			loss += cross_entropy(output, y[i+1])
		return loss

	def evaluate(self,x,vocablist_y,vocabidx_y, device:str):
		e_x = self.dropout(self.encemb(x))
		outenc,(hidden,cell)=self.enclstm(e_x)
		
		y = torch.tensor([vocabidx_y['<cls>']]).to(device)
		pred=[]
		for i in range(30):
			input = y
			input = input.unsqueeze(0)
			input = self.dropout(self.decemb(input))
			outdec,(hidden,cell)= self.declstm(input,(hidden,cell))
			output = self.decout(outdec.squeeze(0))  
			pred_id = output.squeeze().argmax().item()
			if pred_id == vocabidx_y['<eos>']:
				break
			pred_y = vocablist_y[pred_id][0]
			pred.append(pred_y)
			y[0]=pred_id
			input=y
		return pred  

def train(train_data, vocab:Language, epochs:int = 10, 
		  lr:float = 1e-3, device = 'cpu', 
		  model_name:str = 'seq2seq.mode', pre_train:bool = False):

	model = LSTM(vocab.en.get_itos(), vocab.en, vocab.vi.get_itos(), vocab.vi).to(device)
	if pre_train:
		try:
			model.load_state_dict(torch.load(model_name))
			model.eval()
		except RuntimeError:
			print("model not found!")
			return

	optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
	for epoch in range(epochs):
		loss = 0
		for ben, bvi in train_data:
			ben = torch.tensor(ben, dtype=torch.int64).transpose(0,1).to(device) 
			bvi = torch.tensor(bvi, dtype=torch.int64).transpose(0,1).to(device)
			optimizer.zero_grad()
			batchloss = model((ben, bvi),device)
			batchloss.backward()
			optimizer.step() 
			loss = loss + batchloss.item()
		print("epoch", epoch, ": loss", loss)
	torch.save(model.state_dict(), model_name)