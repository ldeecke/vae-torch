import torch
from torch.autograd import Variable
import torch.nn as nn

from itertools import chain

class Encoder(nn.Module):
	"""
	Encoder, x --> mu, log_sigma_sq
	"""
	def __init__(self, config):
		super(Encoder, self).__init__()
		self.config = config
		self.main = nn.Sequential(
			nn.Linear(config.num_channels * config.img_size**2, config.h_dim),
			nn.ReLU()
			)
		self.linear_mu = nn.Linear(self.config.h_dim, self.config.z_dim)
		self.linear_log_sigma_sq = nn.Linear(self.config.h_dim, self.config.z_dim)
		self.reset_bias_and_weights()

	def reset_bias_and_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				m.weight.data.normal_(0.0, self.config.std)
				m.bias.data.normal_(0.0, self.config.std)

	def forward(self, input):
		h = self.main(input)
		return self.linear_mu(h), self.linear_log_sigma_sq(h)


class Decoder(nn.Module):
	"""
	Decoder, N(mu, log_sigma_sq) --> z --> x
	"""
	def __init__(self, config):
		super(Decoder, self).__init__()
		self.config = config
		self.main = nn.Sequential(
			nn.Linear(config.z_dim, config.h_dim),
			nn.ReLU(),
			nn.Linear(config.h_dim, config.num_channels * config.img_size**2),
			nn.Sigmoid()
			)
		self.reset_bias_and_weights()

	def reset_bias_and_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				m.weight.data.normal_(0.0, self.config.std)
				m.bias.data.normal_(0.0, self.config.std)

	def forward(self, input):
		return self.main(input)


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
