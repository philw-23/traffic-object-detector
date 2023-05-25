# %% Load Packages
import tarfile
import torch
import torchvision
import random
import pandas as pd
import os
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader 
from TrafficDetectorDatasets import * # Custom defined classes
from support import *
from PIL import Image

# Disable beta warning
torchvision.disable_beta_transforms_warning()

# %% Extract Data
# Check if extract already exists before extracting
extract_path = './extracted_data/'
if not os.path.exists(extract_path) or len(os.listdir(extract_path)) == 0:
    f = tarfile.open('./object-dataset.tar.gz') # Open file
    f.extractall(extract_path) # Extract to extract location
    
# %% Import Data
# Read in image_data
image_data = pd.read_csv('./extracted_data/object-dataset/labels.csv', sep=' ',
                    names=['frame', 'xmin', 'ymin', 'xmax', 'ymax', 'occluded', 'label', 'light_type'])
image_data['f_path'] = extract_path + 'object-dataset/' + image_data['frame'] # Add full file path
image_data['light_type'] = image_data['light_type'].fillna('No Attribute') # Treat light color/type as attribute
image_data['attribute'] = pd.factorize(image_data['light_type'])[0] # Numeric ID for light type (ignoring for now)
image_data['image_id'] = pd.factorize(image_data['frame'])[0] # Unique ID for frame (accessing "codes" object)
image_data['label_id'] = pd.factorize(image_data['label'])[0] # Unique ID for full label (accessing "codes" object)
label_mapper = dict(zip(image_data['label'], image_data['label_id'])) # For mapping use later
num_classes = len(image_data['label'].unique())

# %% Define Transforms
def get_transform(train):
    transforms = []
    transforms.append(T.ToImageTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    
    return T.Compose(transforms)

train_transforms = get_transform(True)
test_transforms = get_transform(False)

# %% Generate Train/Test Split
train_pct = 0.85 # Percent of data to train on
file_list = image_data['f_path'].unique().tolist() # Unique image list
train_len = int(train_pct * len(file_list)) # Number of training images
random.Random(1).shuffle(file_list) # Shuffle the list
train_files = file_list[:train_len] # Train files
test_files = file_list[train_len:] # Test files
train_frame = image_data.loc[image_data['f_path'].isin(train_files)] # Train df
test_frame = image_data.loc[image_data['f_path'].isin(test_files)] # Test df

# %% Generate/save datasets
tgt_size = (450, 300) # New target image size
train_dataset = TrafficDetectorDataset(train_frame, tgt_size, get_transform(True))
test_dataset = TrafficDetectorDataset(test_frame, tgt_size, get_transform(False))

# %% Create Dataloaders
trainloader = DataLoader(train_dataset, batch_size=2, shuffle=True, 
                         collate_fn=train_dataset.collate_fn)
testloader = DataLoader(test_dataset, batch_size=2, shuffle=False,
                         collate_fn=test_dataset.collate_fn)