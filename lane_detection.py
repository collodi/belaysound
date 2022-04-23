import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence, unpack_sequence

class MyDataset(Dataset):
	def __init__(self, train=True):
		self.xs = torch.load('data/lane_xs')
		self.ys = torch.load('data/lane_ys')

		if train:
			self.idxs = [i for i in range(len(self.ys)) if i % 5 != 0]
		else:
			self.idxs = [i for i in range(len(self.ys)) if i % 5 == 0]

	def __len__(self):
		return len(self.idxs)

	def __getitem__(self, i):
		idx = self.idxs[i]
		return self.xs[idx], self.ys[idx]

class Model(nn.Module):
	def __init__(self):
		super().__init__()

		self.h_dim = 100
		p = 0.2

		self.lstm = nn.LSTM(48, self.h_dim, 1)
		self.fc = nn.Sequential(
				nn.Dropout(p),
				nn.Linear(self.h_dim, self.h_dim),
				nn.ReLU(),
				nn.Dropout(p),
				nn.Linear(self.h_dim, 80),
				nn.ReLU(),
				nn.Dropout(p),
				nn.Linear(80, 60),
				nn.ReLU(),
				nn.Dropout(p),
				nn.Linear(60, 40),
				nn.ReLU(),
				nn.Dropout(p),
				nn.Linear(40, 20),
				nn.ReLU(),
				nn.Dropout(p),
				nn.Linear(20, 10),
				nn.ReLU(),
				nn.Linear(10, 2),
				nn.Softmax(dim=1)
			)

	def postprocess(self, x):
		x_a = F.avg_pool1d(x.t(), 5, stride=1, padding=2).t()
		x_a[:2, :] = x[:2, :]
		x_a[-2:, :] = x[-2:, :]
		return x_a

	def forward(self, x):
		x, _ = self.lstm(x)
		x = unpack_sequence(x)
		x = torch.cat(x, dim=0)

		x = x.view(-1, self.h_dim)
		x = self.fc(x)
		return self.postprocess(x)

def train(m, dev, loader, opt):
	m.train()

	total = len(loader)

	for idx, (x, y) in enumerate(loader):
		x, y = x.to(dev).float(), y.to(dev).long()
		opt.zero_grad()

		y = unpack_sequence(y)
		y = torch.cat(y, dim=0)

		out = m(x)
		loss = F.cross_entropy(out, y)
		loss.backward()

		opt.step()

		perc = 100. * idx / total
		print(f'{perc:.0f}% -> loss: {loss.item():.6f}')

def test_accuracy(m, dev, loader):
	m.eval()

	with torch.no_grad():
		correct = 0
		avg_loss = 0

		cnt = 0

		for idx, (x, y) in enumerate(loader):
			x, y = x.to(dev).float(), y.to(dev).long()

			y = unpack_sequence(y)
			y = torch.cat(y, dim=0)

			out = m(x)

			cnt += len(y)
			correct += sum(out.argmax(dim=1) == y)
			avg_loss += F.cross_entropy(out, y, reduction='sum')

		avg_loss /= cnt
		print(f'Test average loss: {avg_loss:.6f}')

		return correct.float() / cnt

def collate(data):
	xs, ys = [], []
	for x, y in data:
		xs.append(x)
		ys.append(y)

	xs = pack_sequence(xs, enforce_sorted=False)
	ys = pack_sequence(ys, enforce_sorted=False)
	return xs, ys

def main():
	has_cuda = torch.cuda.is_available()

	dev = torch.device('cuda' if has_cuda else 'cpu')
	default_tensor = torch.cuda.FloatTensor if has_cuda else torch.FloatTensor

	torch.set_default_dtype(torch.float32)
	torch.set_default_tensor_type(default_tensor)

	m = Model().to(dev)
	opt = torch.optim.Adam(m.parameters())

	kwargs = { 'pin_memory': True } if has_cuda else {}
	train_loader = DataLoader(MyDataset(train=True), batch_size=5, shuffle=True, collate_fn=collate, **kwargs)
	test_loader = DataLoader(MyDataset(train=False), batch_size=1, shuffle=False, collate_fn=collate, **kwargs)

	m.load_state_dict(torch.load('./lane0.net'))
	acc = test_accuracy(m, dev, train_loader)
	print(f'Train Accuracy: {acc:.6f}')
	acc = test_accuracy(m, dev, test_loader)
	print(f'Test Accuracy: {acc:.6f}')
	return

	nepochs = 30
	for epoch in range(nepochs):
		print(f'=== epoch {epoch + 1}')
		train(m, dev, train_loader, opt)

		acc = test_accuracy(m, dev, train_loader)
		print(f'Train Accuracy: {acc:.6f}')

		acc = test_accuracy(m, dev, test_loader)
		print(f'Test Accuracy: {acc:.6f}')

	torch.save(m.state_dict(), 'lane.net')

if __name__ == '__main__':
	main()
