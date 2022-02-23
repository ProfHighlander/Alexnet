# 2-23-22
# Dataloader for Tiny Image Net

import cv2
import os
import torch
from torch.utils.data import Dataset

class TinyImageDataloader(Dataset):
	def __init__(self, directory, percentile=100):
		super(TinyImageDataloader, self).__init__()
		self.directory = directory # Train directory

		self.percentile = percentile

		self.files, self.boxes = self.parse_data()

	def parse_data(self):
		"""
		Goes through the Tiny Image Net train 
		folder, reads images from the folders, 
		and gets the truths (bounding boxes) 
		for said images
		"""

		files = [] # List of paths to image files
		boxes = [] # List of bouding boxes/truths
		self.line = []
		total_count = 0

		folders = os.listdir(self.directory) # First layer of folders

		# Each type of object in the pictures has its own folder
		for category in folders:
			fh = open(self.directory + "/" + category + "/" + category + "_boxes.txt", "r")
			contents = fh.readlines()

			# Loop through file names in the "index file" 
			# and add those paths to files
			for line in contents:
				
				# Evenly skips files based upon the 
				# percentile to load
				total_count += 1
				if total_count % int(100 / self.percentile) != 0:
					continue
				
				split_line = line.split("\t")
				img_name = split_line[0]
				files.append(self.directory + "/" + category + "/images/" + img_name)
				boxes.append([int(split_line[1]), int(split_line[2]), int(split_line[3]), int(split_line[4])])

		return files, boxes

	def __getitem__(self, idx):
		"""
		Called by the PyTorch dataloader (?), 
		returns a tensor cointaining scaled 
		TinyImageNet images and a tensor with 
		the normalized truths
		"""

		img = cv2.imread(self.files[idx])

		img = cv2.resize(img, (224, 224)) # Scaling
		img = torch.Tensor(img) # Convert to tensor
		img = torch.transpose(img, 1, 2) # Switiching around the order of elements
		img = torch.transpose(img, 0, 1)

		# Normalize truth coordinates
		scaled_boxes = [self.boxes[idx][0] / 63, self.boxes[idx][1] / 63, self.boxes[idx][2] / 63, self.boxes[idx][3] / 63]		

		return img, torch.Tensor(scaled_boxes)

	def __len__(self):
		return len(self.files)