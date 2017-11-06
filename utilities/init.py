import numpy as np
import os
import torch, torchvision

from utilities.data import MNIST, CIFAR10

def init_data_loader(dataset, data_path, batch_size, train=True, training_digits=None):
	if dataset == "mnist":
		if training_digits is not None:
			return MNIST(data_path, batch_size, train=train, condition_on=[training_digits])
		else:
			return MNIST(data_path, batch_size, train=train)

	elif dataset == "cifar10":
		if training_digits is not None:
			return CIFAR10(data_path, batch_size, train=train, condition_on=[training_digits])
		else:
			return CIFAR10(data_path, batch_size, train=train)

def make_dirs(*args):
	for dir in args:
		os.makedirs(dir, exist_ok=True)

def write_preprocessor(config):
	preprocessor = {"dataset": config.dataset, "model": config.model, "condition_on": config.training_digits}
	f = open(config.ckpt_path + "/preprocessor.dat", 'w')
	f.write(str(preprocessor))

