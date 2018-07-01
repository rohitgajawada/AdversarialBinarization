import os
import torch
from torchvision import datasets as dsets
from torchvision import transforms
import utils

"""
Dataloaders for the MNIST dataset
"""
class LoadMNIST():
	'''
	Downloads and loads the MNIST dataset.
	Preprocessing -> Data is normalized in Transforms.
	'''
	def __init__(self, opt):
		kwargs = {
		  'num_workers': opt.workers,
		  'batch_size' : opt.batch_size,
		  'shuffle' : True,
		  'pin_memory': True}

		self.train_loader = torch.utils.data.DataLoader(
			dsets.MNIST('../data', train=True, download=True,
					transform=transforms.Compose([
						transforms.RandomCrop(28, padding=4),
						transforms.RandomHorizontalFlip(),
						transforms.ToTensor(),
						transforms.Normalize((0.1307,), (0.3081,))
					   ])),
			 **kwargs)

		self.val_loader = torch.utils.data.DataLoader(
			dsets.MNIST('../data', train=False,
			  transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize((0.1307,), (0.3081,))
					   ])),
			  **kwargs)

"""
Dataloaders for the CIFAR-10 dataset
"""
class LoadCIFAR10():
	def __init__(self, opt):
		kwargs = {
		  'num_workers': opt.workers,
		  'batch_size' : opt.batch_size,
		  'shuffle' : True,
		  'pin_memory': True}

		self.train_loader = torch.utils.data.DataLoader(
			dsets.CIFAR10('../data', train=True, download=True,
					transform=transforms.Compose([
						transforms.RandomCrop(32, padding=4),
						transforms.RandomHorizontalFlip(),
						transforms.ToTensor(),
						transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
std=[x/255.0 for x in [63.0, 62.1, 66.7]])
					   ])),
			 **kwargs)

		self.val_loader = torch.utils.data.DataLoader(
			dsets.CIFAR10('../data', train=False,
			  transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
std=[x/255.0 for x in [63.0, 62.1, 66.7]])
					   ])),
		  **kwargs)

"""
Dataloaders for the CIFAR-100 dataset
"""
class LoadCIFAR100():
	def __init__(self, opt):
		kwargs = {
		  'num_workers': opt.workers,
		  'batch_size' : opt.batch_size,
		  'shuffle' : True,
		  'pin_memory': True}

		self.train_loader = torch.utils.data.DataLoader(
			dsets.CIFAR100('../data', train=True, download=True,
					transform=transforms.Compose([
						transforms.RandomCrop(32, padding=4),
						transforms.RandomHorizontalFlip(),
						transforms.ToTensor(),
						transforms.Normalize(mean=[x/255.0 for x in [129.3, 124.1, 112.4]],
std=[x/255.0 for x in [68.2, 65.4, 70.4]])
					   ])),
			 **kwargs)

		self.val_loader = torch.utils.data.DataLoader(
			dsets.CIFAR100('../data', train=False,
			  transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize(mean=[x/255.0 for x in [129.3, 124.1, 112.4]],
std=[x/255.0 for x in [68.2, 65.4, 70.4]])
					   ])),
		  **kwargs)
