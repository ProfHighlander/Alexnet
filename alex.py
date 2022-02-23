# 2-23-22
# Menu for the Alexnet model

import torch
from datasets import TinyImageDataloader
from torch import nn
from torch.utils.data import DataLoader

import alex_model as am
import model_tools as mt

	
def load_save(filename):
	"""
	Try to load the specified model
	"""

	try:
		model = torch.load(filename)
		print("Loaded", filename)
	except FileNotFoundError:
		model = am.Alex()
		print(filename, "not found, starting fresh")
	
	return model

def load_dataset(location, percentile, batch_size=300):
	"""
	Loads the TinyImageNet database with a few 
	basic paramters/options
	"""

	imgnet = TinyImageDataloader(location, percentile=percentile)
	data = DataLoader(imgnet, batch_size=batch_size, num_workers=2, shuffle=False, drop_last=True)

	return data

def settings(mode, setting_to_change=0, new_value=""):
	"""
	This function can read and change the user 
	settings. Mode can be 'read' or 'write'. If 
	mode='write' then setting_to_change needs to 
	be set to the index of the setting you want to 
	change in the default_options list. new_value 
	needs to be equal to what the option should be 
	after changing
	"""
	
	# Default model name, percentage of dataset, dataset batch size
	default_options = ["model_save.save", 100, 300]

	# Read options
	if mode == "read":
		try:
			fh = open("settings.txt", 'r')
			options = fh.read()
			options = options.split("\n")
			return options

		except FileNotFoundError:
			print("Settings file not found, creating new")

			# Create default settings
			for x in range(3):
				settings("write", x, default_options[x])
				return default_options
		
	# Change options
	elif mode == "write":
		default_options[setting_to_change] = new_value
		fh = open("settings.txt", 'w+')
		for x in default_options:
			fh.write(str(x) + "\n")
		

# Load important variables
op = ""
model = load_save(settings("read")[0])
percentile = int(settings("read")[1])
batch_size = settings("read")[2]
print("Using", percentile, "% of the dataset")

train_directory = "./tiny-imagenet-200/train"
data = load_dataset(train_directory, percentile)

# Menu
while op != "exit":
	print("1) Train\n2) Test\n3) Visualize\n4) Load\n5) Options\n6) Exit")
	op = input(": ")
	
	if op == "1":
		mt.train_model(model, data, batch_size, 0.001, 0.1)
	elif op == "2":
		mt.test_model(model, data)
	elif op == "3":
		mt.visualize(model, data)
	elif op == "4":
		load_this = input("Load what model? ")
		model = load_save(load_this)
	elif op == "5":
		# Changeable options
		print("1) Default save:", settings("read")[0])
		print("2) Percentile of dataset:", settings("read")[1])
		print("3) Batch size for data:", settings("read")[2])

		# User input
		change_this = input("Change what setting? ")
		change_to = input("Change to what? ")

		# Changes the settings with no error-checking
		settings("write", int(change_this) - 1, change_to)
		print()

		# Resets variables with new preferences
		model = load_save(settings("read")[0])
		percentile = int(settings("read")[1])
		batch_size = settings("read")[2]
		print("Using", percentile, "% of the dataset")
		print()
	elif op == "6":
		op = "exit"

	print()