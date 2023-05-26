# %% Load Packages
import tarfile
import random
import pandas as pd
import os
from torch.utils.data import DataLoader
from models import * # Model/optimizer info
from TrafficDetectorDatasets import * # Custom defined classes
from support import * # Support functions

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
train_dataset = TrafficDetectorDataset(train_frame, tgt_size, get_transforms(True))
test_dataset = TrafficDetectorDataset(test_frame, tgt_size, get_transforms(False))

# %% Create Dataloaders
trainloader = DataLoader(train_dataset, batch_size=2, shuffle=True, 
                         collate_fn=train_dataset.collate_fn)
testloader = DataLoader(test_dataset, batch_size=2, shuffle=False,
                         collate_fn=test_dataset.collate_fn)

# %% Train
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, optimizer, lr_scheduler = get_model_items('retinanet', num_classes + 1, 1, 'SGD')
model.to(device) # Send model to device
num_epochs = 10
train_losses = []
val_losses = []
model_metrics = []

# Execute
for e in range(num_epochs):
    # Train
    epoch_train_loss = train_one_epoch(model, optimizer, trainloader, device)
    train_losses.append(epoch_train_loss) # Log train loss

    # Evaluate
    epoch_val_loss, epoch_metrics = evaluate_one_epoch(model, optimizer, testloader, device)
    val_losses.append(epoch_val_loss) # Log val loss
    model_metrics.append(epoch_metrics) # Log epoch metrics

    # Step scheduler if applicable
    if lr_scheduler:
        lr_scheduler.step(epoch_val_loss)

    # Print epoch results
    print(epoch_train_loss, epoch_val_loss, model_metrics)

