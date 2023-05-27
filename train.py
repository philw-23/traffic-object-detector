# %% Load Packages
import yaml
from torch.utils.data import DataLoader
from models import * # Model/optimizer info
from TrafficDetectorDatasets import * # Custom defined classes
from support import * # Support functions

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
model.to(device) # Send model to device

# Performance tracking
train_losses = []
val_losses = []
model_metrics = []

# Execute
for e in range(NUM_EPOCHS):
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

