import torch
import pandas as pd
import tarfile
import random
import os
import transforms as T
from torchmetrics.detection.mean_ap import MeanAveragePrecision

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
		# f = tarfile.open(dataset_tarfile) # Open file
		# f.extractall(export_path) # Extract to extract location

	# Read in and preprocess data
	image_data = pd.read_csv(export_path + label_file_name, sep=' ',
                    names=['frame', 'xmin', 'ymin', 'xmax', 'ymax', 'occluded', 'label', 'light_type'])
	image_data['f_path'] = export_path + image_data['frame'] # Add full file path
	image_data['light_type'] = image_data['light_type'].fillna('No Attribute') # Treat light color/type as attribute
	image_data['attribute'] = pd.factorize(image_data['light_type'])[0] # Numeric ID for light type (ignoring for now)
	image_data['image_id'] = pd.factorize(image_data['frame'])[0] # Unique ID for frame (accessing "codes" object)
	image_data['label_id'] = pd.factorize(image_data['label'])[0] # Unique ID for full label (accessing "codes" object)
	label_mapper = dict(zip(image_data['label'], image_data['label_id'])) # For mapping use later
	num_classes = len(image_data['label'].unique())

	# Generate dataframes
	file_list = image_data['f_path'].unique().tolist() # Unique image list
	train_len = int(train_pct * len(file_list)) # Number of training images
	random.Random(1).shuffle(file_list) # Shuffle the list
	train_files = file_list[:train_len] # Train files
	test_files = file_list[train_len:] # Test files
	train_frame = image_data.loc[image_data['f_path'].isin(train_files)] # Train df
	test_frame = image_data.loc[image_data['f_path'].isin(test_files)] # Test df

	return train_frame, test_frame, label_mapper, num_classes