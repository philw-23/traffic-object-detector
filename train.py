# %% Load Packages
import yaml
import os
import shutil
import multiprocessing as mp
from torch.utils.data import DataLoader, WeightedRandomSampler
from models import * # Model/optimizer info
from TrafficDetectorDatasets import * # Custom defined classes
from support import * # Support functions
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
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

# %% Generate Weights for train set sampling
# Get class frequencies
class_frequencies = dict(zip(train_frame['label_id'], train_frame['class_counts']))

# Calculate weight for each image
sample_weights = []
for image in train_frame['frame'].unique():
    image_classes = train_frame[train_frame['frame'] == image]['label_id'].unique() # Classes in image
    image_weight = 1.0 / np.sqrt(np.min([class_frequencies[c] for c in image_classes])) # Normalized min of class weights
    sample_weights.append(image_weight) # Append weight for image
    
# Convert to numpy array    
sample_weights = np.array(sample_weights) # Need numpy array

# Generate sampler
image_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),
                                      replacement=True)

# %% Generate/save datasets
train_dataset = TrafficDetectorDataset(train_frame, TGT_SIZE, get_transforms(True))
test_dataset = TrafficDetectorDataset(test_frame, TGT_SIZE, get_transforms(False))

# %% Create Dataloaders
cpu_count = mp.cpu_count()
trainloader = DataLoader(train_dataset, batch_size=1, 
                         sampler=image_sampler, # sampler replaces shuffle
                         num_workers=cpu_count, # Use number of CPU cores for workers
                         collate_fn=train_dataset.collate_fn)
testloader = DataLoader(test_dataset, batch_size=1, 
                        num_workers=cpu_count,
                        shuffle=False,
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

# %% Define model and callbacks
# Best performing model callback
best_performing_callback = ModelCheckpoint( # Log best model
    monitor='val_loss', # Monitor validation loss
    dirpath=location_name, # Location to write checkpoint
    filename='best-model-epoch{epoch:02d}-' + MODEL + '-' + BACKBONE,
    save_top_k=1, # Only save best model
    mode='min'
)

# Current epoch callback
current_epoch_callback = ModelCheckpoint(
    dirpath=location_name,
    filename='current-epoch-checkpoint', # Log most current epoch
    save_top_k=1,
    every_n_epochs=1
)

# Early stopping callback
early_stopper = EarlyStopping( # To prevent overfitting
    monitor='val_loss', # Monitor val loss
    mode='min',
    patience=3 # Three epochs of patience
)
csv_logger = loggers.CSVLogger(location_name, 'log_file')
final_model = LightningModel(model, optimizer)

# %% Define trainer and train model
# Look for checkpoint
if os.path.isfile(location_name + '/current-epoch-checkpoint.ckpt'):
    resume_location = location_name + '/current-epoch-checkpoint.ckpt' # Resume from last completed
else:
    resume_location = None # train from scratch

# Train
trainer = Trainer(max_epochs=NUM_EPOCHS,
                  logger=csv_logger,
                  callbacks=[best_performing_callback, current_epoch_callback, early_stopper])
trainer.fit(final_model, train_dataloaders=trainloader, val_dataloaders=testloader,
            ckpt_path=resume_location)
