import numpy as np
import torch, torchvision
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler


def MNIST(data_path, batch_size, shuffle=True, train=True, condition_on=None, num_workers=0, rescale_to=64, holdout=False):
	img_size, num_channels = 28, 1
	img_size_scaled = rescale_to
	transform = torchvision.transforms.Compose([
								torchvision.transforms.Scale(img_size_scaled),
								torchvision.transforms.CenterCrop(img_size_scaled),
								torchvision.transforms.ToTensor(),
								torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
								])
	dataset = torchvision.datasets.MNIST(data_path, train, download=True, transform=transform)

	if condition_on is not None:
		# sample full dataset once, determine which samples belong to conditioned class
		sampler = SequentialSampler(dataset)
		data_loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=len(dataset), num_workers=num_workers)
		data_iter = iter(data_loader)
		_, labels = data_iter.next()
		ids = np.where(np.in1d(labels.numpy().ravel(), condition_on))[0]

		if not holdout:
			# sample randomly without replacement from conditioned class
			sampler = SubsetRandomSampler(ids)
			return torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers), img_size_scaled, num_channels

		else:
			split = int(0.9 * len(ids))
			ids_train, ids_holdout = ids[:split], ids[split:]
			sampler_train = SubsetRandomSampler(ids_train)
			sampler_holdout = SubsetRandomSampler(ids_holdout)
			return torch.utils.data.DataLoader(dataset, sampler=sampler_train, batch_size=batch_size, num_workers=num_workers), torch.utils.data.DataLoader(dataset, sampler=sampler_holdout, batch_size=batch_size, num_workers=num_workers), img_size_scaled, num_channels

	else:
		if not holdout:
			return torch.utils.data.DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers), img_size_scaled, num_channels

		else:
			ids = np.arange(0, len(dataset))
			split = int(0.9 * len(ids))
			ids_train, ids_holdout = ids[:split], ids[split:]
			sampler_train = SubsetRandomSampler(ids_train)
			sampler_holdout = SubsetRandomSampler(ids_holdout)
			return torch.utils.data.DataLoader(dataset, sampler=sampler_train, batch_size=batch_size, num_workers=num_workers), torch.utils.data.DataLoader(dataset, sampler=sampler_holdout, batch_size=batch_size, num_workers=num_workers), img_size_scaled, num_channels


def CIFAR10(data_path, batch_size, shuffle=True, train=True, condition_on=None, num_workers=0, rescale_to=64, holdout=False):
	img_size, num_channels = 32, 3
	img_size_scaled = rescale_to
	transform = torchvision.transforms.Compose([
								torchvision.transforms.Scale(img_size_scaled),
								torchvision.transforms.CenterCrop(img_size_scaled),
								torchvision.transforms.ToTensor(),
								torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
								])
	dataset = torchvision.datasets.CIFAR10(data_path, train, download=True, transform=transform)

	if condition_on is not None:
		# sample full dataset once, determine which samples belong to conditioned class
		sampler = SequentialSampler(dataset)
		data_loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=len(dataset), num_workers=num_workers)
		data_iter = iter(data_loader)
		_, labels = data_iter.next()
		ids = np.where(np.in1d(labels.numpy().ravel(), condition_on))[0]

		if not holdout:
			# sample randomly without replacement from conditioned class
			sampler = SubsetRandomSampler(ids)
			return torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers), img_size_scaled, num_channels

		else:
			split = int(0.9 * len(ids))
			ids_train, ids_holdout = ids[:split], ids[split:]
			sampler_train = SubsetRandomSampler(ids_train)
			sampler_holdout = SubsetRandomSampler(ids_holdout)
			return torch.utils.data.DataLoader(dataset, sampler=sampler_train, batch_size=batch_size, num_workers=num_workers), torch.utils.data.DataLoader(dataset, sampler=sampler_holdout, batch_size=batch_size, num_workers=num_workers), img_size_scaled, num_channels

	else:
		if not holdout:
			return torch.utils.data.DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers), img_size_scaled, num_channels

		else:
			ids = np.arange(0, len(dataset))
			split = int(0.9 * len(ids))
			ids_train, ids_holdout = ids[:split], ids[split:]
			sampler_train = SubsetRandomSampler(ids_train)
			sampler_holdout = SubsetRandomSampler(ids_holdout)
			return torch.utils.data.DataLoader(dataset, sampler=sampler_train, batch_size=batch_size, num_workers=num_workers), torch.utils.data.DataLoader(dataset, sampler=sampler_holdout, batch_size=batch_size, num_workers=num_workers), img_size_scaled, num_channels


def SVHN(data_path, batch_size, shuffle=True, split="train", num_workers=0, rescale_to=64):
	img_size, num_channels = 32, 3
	img_size_scaled = rescale_to
	transform = torchvision.transforms.Compose([
								torchvision.transforms.Scale(img_size_scaled),
								torchvision.transforms.CenterCrop(img_size_scaled),
								torchvision.transforms.ToTensor(),
								torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
								])
	dataset = torchvision.datasets.SVHN(data_path, split, download=True, transform=transform)
	return torch.utils.data.DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers), img_size_scaled, num_channels
