from typing import Any
import os
import torch
import pytorch_lightning as pl
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models import get_model
from torchvision.models.detection import RetinaNet, FasterRCNN, backbone_utils
from torchvision.models.detection.rpn import AnchorGenerator
from torchmetrics.detection.mean_ap import MeanAveragePrecision

base_models = {
    'RetinaNet':'retinanet_resnet50_fpn_v2',
    'SSD':'ssd300_vgg16',
    'FasterRCNN':'fasterrcnn_resnet50_fpn_v2'
}

class LightningModel(pl.LightningModule):
    def __init__(self, model, optimizer, log_file):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.log_file = log_file
        self.map_metric = MeanAveragePrecision()
    
    def training_step(self, batch, batch_idx):
        # Execute train step
        images, targets = batch
        images = [image for image in images]
        targets = [{k:v for k, v in t.items()} for t in targets] # list of dicts
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log('train_loss', losses, on_step=False, on_epoch=True) 

        # Return
        return losses
    
    def validation_step(self, batch, batch_idx):
        # Set to train to calcualte loss
        self.model.train()
        images, targets = batch
        images = [image for image in images]
        targets = [{k:v for k, v in t.items()} for t in targets] # list of dicts
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log('val_loss', losses, on_step=False, on_epoch=True)

        # Evaluate
        self.model.eval()
        val_outputs = self.model(images)
        self.map_metric.update(val_outputs, targets)

    def on_validation_epoch_end(self):
        # Compute metrics
        metrics = self.map_metric.compute()

        # Write to log file
        if os.path.isfile(self.log_file): # Exists, we want to append
            write_mode = 'a' # Append
        else:
            write_mode = 'w' # Create
            
        with open(self.log_file, write_mode) as out:
            out.write(f'epoch={self.current_epoch}\t')
            for k, v in self.trainer.logged_metrics.items(): # Losses
                out.write(f'{k}={v}\t')
            for k, v in metrics.items(): # mAP metrics
                out.write(f'{k}={v}\t')
            out.write('\n')

        # Reset metric/epoch log
        self.map_metric.reset()

    def configure_optimizers(self) -> Any:
        return self.optimizer

def get_model_items(model_id, num_classes, optimizer, trainable_backbone_layers=1, 
                    sub_backbone=False, backbone_id=None, backbone_out_channels=None):
    
    model = None # For tracking backbone substitution
    if sub_backbone:

        if model_id == 'SSD':
            print('Backbone sub for SSD unavailable - continuing with default')
            pass

        else:
            try: # Try to develop model with the backbone        
                backbone_model = get_backbone(backbone_id, backbone_out_channels) # Get backbone
                
                # Note: Using default anchor generator for each model class
                if model_id == 'RetinaNet':
                    model = RetinaNet(backbone=backbone_model, 
                                        num_classes=num_classes)

                else: # For FasterRCNN - also need to define roi_pooler
                    model = FasterRCNN(backbone=backbone_model, 
                                       num_classes=num_classes)

            except:
                print('Unable to substitute backbone - preceeding with default')
                pass

    if not sub_backbone or not model: # Handle other cases
        model = get_model(base_models[model_id], 
                        weights=None, # Tops layer untrained
                        num_classes=num_classes,
                        weights_backbone='DEFAULT', # Always start with pretrained backbone
                        trainable_backbone_layers=trainable_backbone_layers)
    
    # Get parameters to optimizer
    params = [p for p in model.parameters() if p.requires_grad] # trainable params

    if optimizer == 'sgd': # Stochastic gradient descent
        optimizer = torch.optim.SGD(params, lr=1e-2,
                                    momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( # Reduce LR if we aren't making progress
            optimizer, mode='min', patience=2, cooldown=1, min_lr=1e-6
        )

    elif optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=0.001)
        lr_scheduler = None

    return model, optimizer, lr_scheduler
    
def get_backbone(backbone_id, backbone_out_channels):

    if 'resnet' in backbone_id:
        backbone_model = backbone_utils.resnet_fpn_backbone(backbone_id, 
                                                            weights='DEFAULT') # Pretrained Backbone
        return backbone_model

    return # No backbone substitution available