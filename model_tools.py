# 2-23-22
# Functions used to manipulate Alexnet model

import cv2
import random
import torch
from torch import nn


import intounion as intu

def train_model(model, data, epochs, learn_rate=0.1, momentum=0.1):
	"""
	Used to train the Alexnet model ;)
	"""

	loss = nn.MSELoss() # Loss function
	optim = torch.optim.SGD(model.parameters(), lr=learn_rate, momentum=momentum) # Optimization algorithm
	model.train() # Allows the model to learn

	loss_avg = 0
	counter = 0 # Count

	for epoch in range(epochs):
		
		# Training loop
		for img, truth in data:
			optim.zero_grad() #zero the parameter gradients
			output = model(img) # Used for testing
			
			# Loss calculation
			loss_num = loss(output, truth) # Compare answer to real answer
			loss_num.backward() # Backward optimize
			optim.step()
			loss_avg += loss_num.data # Loss calculations

		counter += 1
		torch.save(model, "model_save.save")
		print("Completed epoch", counter, "out of", epochs)
		print("Saved to: model_save.save!")

			
	return

def test_model(model, data):
	"""
	Tests how accurate the model 
	is at identifying things in 
	the dataset
	"""

	model.eval()
	right = 0
	total = 0
	accuracy_list = []
	x = 0
	for img, truth in data: # (Total # of images / batch size)
		output = model(img)
		 # Calculate accuracy for one pass
		current_accuracy = intu.calc_intersection(truth[0].tolist(), output[0].tolist())
		accuracy_list.append(current_accuracy)
		if x % 10 == 0:
			print((x / len(data) * 100), "%")
		x += 1
	
	# Average the accuracies
	accuracy = sum(accuracy_list) / len(accuracy_list)
	print(accuracy)

def visualize(model, data):
	"""
	Takes in a model and a dataset, 
	picks a random image from the 
	dataset, runs the image through 
	the model, and displays the result
	"""
	
	for img, truth in data:
		# Pick random image
		img_choice = random.randint(0, data.batch_size)
		
		# Run image through model
		output = model(img)
		output = output.tolist()

		# Create window and show image
		cv2.startWindowThread() 
		cv2.namedWindow("Winky face")
		image = img[img_choice].permute(1, 2, 0).numpy() / 224
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

		# Truth square coordinates
		start_tuple = (int(output[img_choice][0] * 224), int(output[img_choice][1] * 224))
		end_tuple = (int(output[img_choice][2] * 224), int(output[img_choice][3] * 224))
		
		tlist = truth.tolist()
		tstart_tuple = (int(tlist[img_choice][0] * 224), int(tlist[img_choice][1] * 224))
		tend_tuple = (int(tlist[img_choice][2] * 224), int(tlist[img_choice][3] * 224))

		guess_color = (255, 255, 0) # Blue
		truth_color = (0, 0, 255) # Red

		# Print the IOU for the image and guess
		print("IOU:", intu.calc_intersection(truth[img_choice].tolist(), output[img_choice]))

		# Combine image and guess rectangle
		image_and_guess = cv2.rectangle(image, start_tuple, end_tuple, color=guess_color, thickness=3)

		# Add truth rectangle to image and guess rectangle
		image_guess_truth = cv2.rectangle(image_and_guess, tstart_tuple, tend_tuple, color=truth_color, thickness=3)

		# Show result
		cv2.imshow("Winky face", image_guess_truth)
		cv2.waitKey()
		cv2.destroyAllWindows()
		cv2.waitKey(1)
		
		return
