import torch
from torch.autograd import Variable
import torchvision

import numpy as np
import os, time

from vae.parser import get_outlier_parser, update_img_and_filter_dims
from vae.data import CIFAR10, MNIST, SVHN
from vae.nn import VAE

to_np = lambda x: x.data.cpu().numpy()
normalize_to_zero_one = lambda x: (x + 1.) / 2.

if __name__ == "__main__":

	torch.backends.cudnn.benchmark = True

	parser = get_outlier_parser()
	config = parser.parse_args()

	os.makedirs(config.ckpt_path, exist_ok=True)
	os.makedirs(config.data_path, exist_ok=True)

	data_loader, img_size, num_channels = CIFAR10(config.data_path, 1, config.num_workers, train=False)
	update_img_and_filter_dims(config, img_size, num_channels)

	v = VAE(config)
	v = v.cuda()
	v.load_state_dict(torch.load(os.path.join(config.ckpt_path, 'v.pth')))

	v_optim = torch.optim.Adam(v.parameters(), lr=config.lr_adam, betas=(config.beta_1, config.beta_2))

	bce_loss = torch.nn.BCELoss(size_average=False) # no averaging, sum over batch, height and width

	storage = {"labels": [], "losses": []}

	for step, (images, labels) in enumerate(data_loader, 0):
		if step >= 1: print("[%i] [%2.2f]" % (step, time.time() - t1), end="\r")
		t1 = time.time()

		batch_size = images.size(0)

		x = Variable(images.type(torch.cuda.FloatTensor))
		# x = x.resize(batch_size, config.num_channels * config.img_size**2)
		x = x.repeat(config.num_searches, 1, 1, 1)
		x = normalize_to_zero_one(x)
		x_r = v(x) # reconstruction

		loss_r = bce_loss(x_r, x) / batch_size
		loss_kl = torch.mean(.5 * torch.sum((v.mu**2) + torch.exp(v.log_sigma_sq) - 1 - v.log_sigma_sq, 1))
		loss = loss_r + loss_kl

		storage["labels"].append(*labels.numpy())
		storage["losses"].append(*to_np(loss))

		if step >= config.num_test_samples: break

	np.savetxt("labels.dat", np.asarray(storage["labels"]), delimiter=",", fmt=["%i"], comments="", header="")
	np.savetxt("losses.dat", np.asarray(storage["losses"]), delimiter=",", fmt=["%2.8e"], comments="", header="")
