import torch
from torch.autograd import Variable
import torchvision

import numpy as np
import ast, os, time

from utilities.data import CIFAR10, MNIST, SVHN
from utilities.init import make_dirs, init_data_loader
from utilities.output import write_logger, write_observations
from utilities.parser import get_default_parser, extend_parser_with_outlier_task, update_code_dim
from architecture.nn import VAE

to_np = lambda x: x.data.cpu().numpy()
normalize_to_zero_one = lambda x: (x + 1.) / 2.

if __name__ == "__main__":

	torch.backends.cudnn.benchmark = True

	parser = get_default_parser()
	parser = extend_parser_with_outlier_task(parser)
	config = parser.parse_args()

	# determine type of checkpointed model, and dataset it was trained on
	with open(os.path.join(config.ckpt_path, "preprocessor.dat"), 'r') as f:
		preprocessor = ast.literal_eval(f.read())
		print(preprocessor)

	make_dirs(config.ckpt_path, config.data_path)
	if config.output_path is not "":
		make_dirs(config.output_path)

	data_loader, config.img_size, config.num_channels = init_data_loader(preprocessor["dataset"], config.data_path, 1, train=False)
	update_code_dim(config)

	v = VAE(config)
	v = v.cuda()
	v.load_state_dict(torch.load(os.path.join(config.ckpt_path, 'v.pth')))
	v.eval()

	bce_loss = torch.nn.BCELoss(size_average=False) # no averaging, sum over batch, height and width

	storage = {"labels": [], "losses": []}

	for step, (images, labels) in enumerate(data_loader, 0):
		if step >= 1: print("[%i] [%2.2f]" % (step, time.time() - t1), end="\r")
		t1 = time.time()

		batch_size = images.size(0)

		x = Variable(images.type(torch.cuda.FloatTensor))
		x = x.repeat(config.num_searches, 1, 1, 1)
		x = normalize_to_zero_one(x)
		x_r = v(x) # reconstruction

		loss_r = bce_loss(x_r, x) / batch_size
		loss_kl = torch.mean(.5 * torch.sum((v.mu**2) + torch.exp(v.log_sigma_sq) - 1 - v.log_sigma_sq, 1))
		loss = loss_r + loss_kl

		storage["labels"].append(*labels.numpy())
		storage["losses"].append(*to_np(loss))

		if step >= config.num_test_samples:
			break

	np.savetxt(os.path.join(config.output_path, "labels.dat"),
		np.asarray(storage["labels"]),
		delimiter=",",
		fmt=["%i"],
		comments="",
		header="")
	np.savetxt(os.path.join(config.output_path, "losses.dat"),
		np.asarray(storage["losses"]),
		delimiter=",",
		fmt=["%2.8e"],
		comments="",
		header="")
