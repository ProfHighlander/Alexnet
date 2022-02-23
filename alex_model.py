# 2-1-22
# Cal Tipton
# A reproduction of the Alexnet neural network

import torch
from torch import nn
from numpy import DataSource

class Alex(nn.Module):
	"""
	Neural netowrk with almost identical 
	specs to the original Alexnet model
	""" 

	def __init__(self):
		super().__init__()
		self.first_conv = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)
		self.second_conv = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5)
		self.pool_one = nn.MaxPool2d(kernel_size=3, stride=2)
		self.third_conv = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3)
		self.fourth_conv = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3)
		self.fifth_conv = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3)

		self.sixth_fully = nn.Linear(256 * 4 * 4, 4096)
		self.seventh_fully = nn.Linear(4096, 4096)
		self.eighth_fully = nn.Linear(4096, 4)
		self.log_soft = nn.LogSoftmax(dim=1)

		self.learn_rate = 0.9
		self.momentum = 0.9

	def forward(self, x):
		x = self.first_conv(x)
		x = torch.relu(x)
		x = self.pool_one(x)

		x = self.second_conv(x)
		x = torch.relu(x)
		x = self.pool_one(x)

		x = self.third_conv(x)
		x = torch.relu(x)

		x = self.fourth_conv(x)
		x = torch.relu(x)

		x = self.fifth_conv(x)
		x = torch.relu(x)

		x = x.view(x.size(0), 256*4*4)

		x = self.sixth_fully(x)
		x = torch.relu(x)

		x = self.seventh_fully(x)
		x = torch.relu(x)

		x = self.eighth_fully(x)
		x = torch.relu(x)

		return x

