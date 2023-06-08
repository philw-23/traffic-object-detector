# %% Load Packages
import yaml
import os
import shutil
from torch.utils.data import DataLoader
from models import * # Model/optimizer info
from TrafficDetectorDatasets import * # Custom defined classes
from support import * # Support functions
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from datetime import datetime

# %% Read in parameters
with open('./parameters.yml', 'r') as f:
    parameters = yaml.safe_load(f)

# Model parameters
MODEL = parameters['MODEL']
BACKBONE_SUB = parameters['BACKBONE_SUB']
BACKBONE = parameters['BACKBONE']
BACKBONE_OUT_CHANNELS = parameters['BACKBONE_OUT_CHANNELS']
OPTIMIZER = parameters['OPTIMIZER']

# Data parameters
IM_WIDTH = parameters['TGT_SIZE'][0]
IM_HEIGHT = parameters['TGT_SIZE'][1]
TGT_SIZE = (IM_WIDTH, IM_HEIGHT)
PCT_TRAIN = parameters['PCT_TRAIN']
TARFILE = parameters['TARFILE']
EXPORT_PATH = parameters['EXPORT_PATH'] # Where data is extracted to
LABEL_FILE = parameters['LABEL_FILE']

# Training parameters
NUM_EPOCHS = parameters['NUM_EPOCHS']
BASE_TRAIN_LOC = parameters['BASE_TRAIN_LOC']


# %% Extract Data
train_frame, test_frame, label_mapper, num_classes = extract_data(EXPORT_PATH, TARFILE,
                                                                  LABEL_FILE, PCT_TRAIN)
    
# %% Generate/save datasets
train_dataset = TrafficDetectorDataset(train_frame, TGT_SIZE, get_transforms(True))
test_dataset = TrafficDetectorDataset(test_frame, TGT_SIZE, get_transforms(False))

# %% Create Dataloaders
trainloader = DataLoader(train_dataset, batch_size=1, shuffle=True, 
                         collate_fn=train_dataset.collate_fn)
testloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                         collate_fn=test_dataset.collate_fn)

# %% Train
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, optimizer, lr_scheduler = get_model_items(MODEL, num_classes + 1, OPTIMIZER,
                                                 trainable_backbone_layers=1,
                                                 sub_backbone=BACKBONE_SUB,
                                                 backbone_id =BACKBONE,
                                                 backbone_out_channels=BACKBONE_OUT_CHANNELS)

# Set location for writing outputs
current_time = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
location_name = BASE_TRAIN_LOC + '/' + MODEL + '_' + BACKBONE + '_' + current_time
if not os.path.exists(location_name): # Create directory if it doesn't exist
    os.makedirs(location_name)
    shutil.copy('./parameters.yml', location_name + '/' + 'parameters.yml')

# Define lightning model, trainer, csv logger, callback
checkpoint_log = ModelCheckpoint( # Log best model
    monitor='val_loss', # Monitor validation loss
    dirpath=location_name, # Location to write checkpoint
    filename='best-model-epoch{epoch:02d}-' + MODEL + '-' + BACKBONE,
    save_top_k=1, # Only save best model
    mode='min'
)
csv_logger = loggers.CSVLogger(location_name, 'log_file')
final_model = LightningModel(model, optimizer)
trainer = Trainer(max_epochs=1,
                  logger=csv_logger,
                  callbacks=[checkpoint_log])

# Train using lightning model
trainer.fit(final_model, train_dataloaders=trainloader, val_dataloaders=testloader)
