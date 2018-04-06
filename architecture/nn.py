import torch
from torch.autograd import Variable
import torch.nn as nn

from itertools import chain
from numpy import prod


class Encoder(nn.Module):
	"""
	Encoder, x --> mu, log_sigma_sq
	"""
	def __init__(self, config):
		super(Encoder, self).__init__()
		self.config = config

		self.main = nn.Sequential(
			nn.Conv2d(config.num_channels, 64, 4, 2, 1, bias=False),
			nn.ReLU(True),
			nn.BatchNorm2d(64),
			nn.Conv2d(64, 128, 4, 2, 1, bias=False),
			nn.ReLU(True),
			nn.BatchNorm2d(128),
			nn.Conv2d(128, 256, 4, 2, 1, bias=False),
			nn.ReLU(True),
			nn.BatchNorm2d(256),
			nn.Conv2d(256, 128, 4, 2, 1, bias=False),
			nn.ReLU(True),
			nn.BatchNorm2d(128)
			)
		self.linear_mu = nn.Linear(int(prod(config.c_dim)), self.config.z_dim)
		self.linear_log_sigma_sq = nn.Linear(int(prod(config.c_dim)), self.config.z_dim)
		self.reset_bias_and_weights()

	def reset_bias_and_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				m.weight.data.normal_(0.0, self.config.std)
				m.bias.data.normal_(0.0, self.config.std)
			elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
				m.weight.data.normal_(0.0, self.config.std)
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.normal_(1.0, self.config.std)
				m.bias.data.zero_()

	def forward(self, input):
		h = self.main(input)
		h = h.resize(h.size(0), h.size(1) * h.size(2) * h.size(3))
		return self.linear_mu(h), self.linear_log_sigma_sq(h)


class Decoder(nn.Module):
	"""
	Decoder, N(mu, log_sigma_sq) --> z --> x
	"""
	def __init__(self, config):
		super(Decoder, self).__init__()
		self.config = config
		self.main_1 = nn.Sequential(
			nn.Linear(config.z_dim, int(prod(config.c_dim))),
			nn.ReLU(),
			nn.Linear(int(prod(config.c_dim)), int(prod(config.c_dim))),
			nn.ReLU()
			)
		self.main_2 = nn.Sequential(
			nn.ConvTranspose2d(128, 256, 4, 2, 1, bias=False),
			nn.ReLU(True),
			nn.BatchNorm2d(256),
			nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
			nn.ReLU(True),
			nn.BatchNorm2d(128),
			nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
			nn.ReLU(True),
			nn.BatchNorm2d(64),
			nn.ConvTranspose2d(64, config.num_channels, 4, 2, 1, bias=False),
			nn.Sigmoid()
			)
		self.reset_bias_and_weights()

	def reset_bias_and_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				m.weight.data.normal_(0.0, self.config.std)
				m.bias.data.normal_(0.0, self.config.std)
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
				m.weight.data.normal_(0.0, self.config.std)

	def forward(self, input):
		h = self.main_1(input)
		h = h.resize(input.size(0), *self.config.c_dim)
		x = self.main_2(h)
		return x


class VAE(nn.Module):
	"""
	VAE, x --> mu, log_sigma_sq --> N(mu, log_sigma_sq) --> z --> x
	"""
	def __init__(self, config):
		super(VAE, self).__init__()
		self.config = config
		self.Encoder = Encoder(config)
		self.Decoder = Decoder(config)

	def parameters(self):
		return chain(self.Encoder.parameters(), self.Decoder.parameters())

	def sample_from_q(self, mu, log_sigma_sq):
		epsilon = Variable(torch.randn(mu.size()), requires_grad=False).type(torch.cuda.FloatTensor)
		sigma = torch.exp(log_sigma_sq / 2)
		return mu + sigma * epsilon

	def forward(self, input):
		self.mu, self.log_sigma_sq = self.Encoder(input)
		z = self.sample_from_q(self.mu, self.log_sigma_sq)
		return self.Decoder(z)
