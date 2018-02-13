import torch
from torch.autograd import Variable
import torchvision

import numpy as np
import os, time

from utilities.data import CIFAR10, MNIST, SVHN
from utilities.init import make_dirs, init_data_loader, write_preprocessor
from utilities.output import write_logger, write_observations
from utilities.parser import get_default_parser, update_code_dim
from architecture.nn import VAE

to_np = lambda x: x.data.cpu().numpy()
normalize_to_zero_one = lambda x: (x + 1.) / 2.

if __name__ == "__main__":

	torch.backends.cudnn.benchmark = True

	parser = get_default_parser()
	config = parser.parse_args()

	make_dirs(config.ckpt_path, config.data_path, config.img_path)

	data_loader, config.img_size, config.num_channels = init_data_loader(config.dataset, config.data_path, config.batch_size, training_digits=config.training_digits)

	update_code_dim(config)
	write_preprocessor(config)

	v = VAE(config)
	v = v.cuda()

	if config.ckpt:
		v.load_state_dict(torch.load(os.path.join(config.ckpt_path, 'v.pth')))

	v_optim = torch.optim.Adam(v.parameters(), lr=config.lr_adam, betas=(config.beta_1, config.beta_2))
	bce_loss = torch.nn.BCELoss(size_average=False) # no averaging, sum over batch, height and width

	logger = {"loss": np.array([])}
	z_fixed = Variable(torch.randn(config.num_samples, config.z_dim)).type(torch.cuda.FloatTensor)

	for epoch in range(config.num_epochs):
		if epoch >= 1:
			print("\n[%2.2f]" % (time.time() - t0), end="\n")
		t0 = time.time()

		for step, (images, _) in enumerate(data_loader, 0):
			if step >= 1:
				print("[%i] [%i] [%2.2f] [%2.2f]" % (epoch, step, time.time() - t1, to_np(loss)), end="\r")
			t1 = time.time()

			batch_size = images.size(0)

			x = Variable(images.type(torch.cuda.FloatTensor))
			x = normalize_to_zero_one(x)
			x_r = v(x) # reconstruction

			loss_r = bce_loss(x_r, x) / batch_size
			loss_kl = torch.mean(.5 * torch.sum((v.mu**2) + torch.exp(v.log_sigma_sq) - 1 - v.log_sigma_sq, 1))
			loss = loss_r + loss_kl
			loss.backward()

			v_optim.step()
			v_optim.zero_grad()

		if epoch % config.num_every_nth_epoch == 0:
			write_observations(config, epoch, z_fixed, v, logger, loss)

	write_logger(config, logger)
