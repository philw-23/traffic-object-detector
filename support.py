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
	
	return T.Compose(transforms)
	

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