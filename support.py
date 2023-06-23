import torch
import pandas as pd
import tarfile
import random
import os
import transforms as T
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from sklearn.model_selection import train_test_split

# Gets transformations to apply to images/targets
def get_transforms(train):
	transforms = []
	transforms.append(T.PILToTensor())
	transforms.append(T.ConvertImageDtype(torch.float))
	if train:
		transforms.append(T.RandomHorizontalFlip(0.5))
	transforms.append(T.Normalize(
		[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
	))
	
	return T.Compose(transforms)
	
# Train a single epoch - currently pseudocode but will be updated!
def train_one_epoch(model, optimizer, trainloader, device):
	# Set model to train mode
	model.train()

	# Iterate through data loader
	train_loss = 0 # Total loss for epoch
	for images, boxes, labels in trainloader:
		# Send items to device
		images = [image.to(device) for image in images]
		targets = [{'boxes':boxes[i].to(device), 'labels':labels[i].to(device)} 
					for i in range(len(boxes))] # Send targets to device in list of dictionary form

		# Train
		losses = model(images, targets) # Model returns dictionary of losses!
		batch_loss = sum(loss for loss in losses.values()) # For backward calc
		train_loss += batch_loss # Sum loss for batch
		# Potentially calculate other metrics (accuracy, etc.)

		# Backpropogate
		optimizer.zero_grad()
		batch_loss.backward()
		optimizer.step()

		break

	epoch_loss = train_loss / len(trainloader) # Estimate epoch loss
	return epoch_loss

def evaluate_one_epoch(model, optimizer, testloader, device):
	metric = MeanAveragePrecision() # Will define map metric here
	val_loss = 0 # Total loss for epoch
	with torch.no_grad(): # No updates in evaluation
		for images, boxes, labels in testloader:
			#### TRAIN PASS - to get loss ####
			model.train()
			images = [image.to(device) for image in images]
			targets = [{'boxes':boxes[i].to(device), 'labels':labels[i].to(device)} 
						for i in range(len(boxes))]
			losses = model(images, targets) # Model returns dictionary of losses!
			val_loss += sum(loss for loss in losses.values())
			# Potentially calculate other metrics (accuracy, etc.)

			#### VAL PASS - for evaluating and calculating metrics ####
			model.eval()
			val_outputs = model(images) # Returns predictions in eval mode
			metric.update(val_outputs, targets) # will update map

			break

	eval_results = metric.compute() # Calculate metrics after completing all batches
	epoch_loss = val_loss / len(testloader) # Estimate epoch loss

	return epoch_loss, eval_results

# Function for extracting data to local directory for use
# Function assumes labels in csv format in extracted data folder
# NOTE: currently function is fixed with this dataset, would like to update that in the future
def extract_data(export_path, dataset_tarfile, label_file_name, train_pct=0.8):

	if not os.path.exists(export_path) or len(os.listdir(export_path)) == 0:
		tar = tarfile.open(dataset_tarfile)
		for member in tar.getmembers():
			if member.isfile(): # Skip non file items
				member.name = os.path.basename(member.name) # Remove path
				tar.extract(member, export_path)

	image_data = pd.read_csv(export_path + label_file_name, sep=' ',
					names=['frame', 'xmin', 'ymin', 'xmax', 'ymax', 'occluded', 'label', 'light_type'])
	image_data['f_path'] = export_path + image_data['frame'] # Add full file path
	image_data['label_adjust'] = image_data['label'] + image_data['light_type'].fillna('') # Final labels
	image_data['label_id'] = pd.factorize(image_data['label_adjust'])[0] # Numeric representation
	image_data['class_counts'] = image_data['label_id'].map(image_data['label_id'].value_counts()) # Add class counts for weighting purposes
	min_classes = image_data.loc[image_data.groupby('frame').class_counts.idxmin()] # Get min classes for each image

	# Generate Train/Test split
	train_idxs, test_idxs = train_test_split(
		min_classes['frame'], test_size=0.15,
		stratify=min_classes['label_id']
	)

	# Generate train/test frames
	train_frame = image_data[image_data['frame'].isin(train_idxs)]
	test_frame = image_data[image_data['frame'].isin(test_idxs)]

	# Label mapper/num classes for later
	label_mapper = dict(zip(image_data['label_adjust'], image_data['label_id'])) # For mapping use later
	num_classes = len(image_data['label_id'].unique())

	return train_frame, test_frame, label_mapper, num_classes