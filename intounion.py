# 2-23-22
# Calculates the intersection over union (wow!)

def calc_intersection(first, second):
	"""
	Calculates how overlapped two regions 
	are
	Inputs: Two lists representing squares. 
	Should be [top left X, top left Y, bottom 
	right X, bottom right Y] for both
	"""
	
	# Coordinates for the intersection rectangle
	xa = max(first[0], second[0]) # Top left X
	ya = max(first[1], second[1]) # Top left Y
	xb = min(first[2], second[2]) # Bottom right X
	yb = min(first[3], second[3]) # Bottom left Y
	
	 # Area of intersection
	inter_area = max(0, xb - xa) * max(0, yb - ya)

	# Union (total area of both squares)
	first_area = (first[2] - first[0]) * (first[3] - first[1])
	second_area = (second[2] - second[0]) * (second[3] - second[1])

	iou = inter_area / float(first_area + second_area - inter_area)

	return iou