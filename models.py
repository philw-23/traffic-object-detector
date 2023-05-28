from typing import Any
import torch
import pytorch_lightning as pl
from torchvision.models import get_model
from torchvision.models.detection import RetinaNet, FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchmetrics.detection.mean_ap import MeanAveragePrecision

base_models = {
    'RetinaNet':'retinanet_resnet50_fpn_v2',
    'SSD':'ssd300_vgg16',
    'FasterRCNN':'fasterrcnn_resnet50_fpn_v2'
}

class LightningModel(pl.LightningModule):
    def __init__(self, model, optimizer):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.map_metric = MeanAveragePrecision()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, boxes, labels = batch
        images = [image for image in images]
        targets = [{'boxes':boxes[i], 'labels':labels[i]} for i in range(len(labels))]
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log('train_loss', losses, on_step=False, on_epoch=True) # Log loss on epoch end
        return losses
    
    def validation_step(self, batch, batch_idx):
        # Set to train to calcualte loss
        self.model.train()
        images, boxes, labels = batch
        images = [image for image in images]
        targets = [{'boxes':boxes[i], 'labels':labels[i]} for i in range(len(labels))]
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log('val_loss', losses)

        # Evaluate
        self.model.eval()
        val_outputs = self.model(images)
        self.map_metric(val_outputs, targets)

    def on_validation_epoch_end(self):
        metrics = self.map_metric.compute()
        for k, v in metrics.items():
            self.log(k, v.item())

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
                backbone_model = get_backbone(backbone_id, backbone_out_channels)
                # NOTE: Need to define anchor generator for general model class calls to work
                anchor_generator = AnchorGenerator( 
                    sizes=((32, 64, 128, 256, 512),),
                    aspect_ratios=((0.5, 1.0, 2.0),)
                )

                if model_id == 'RetinaNet':
                    model = RetinaNet(backbone=backbone_model, 
                                        num_classes=num_classes,
                                        anchor_generator=anchor_generator)

                else:
                    model = FasterRCNN(backbone=backbone_model, 
                                        num_classes=num_classes,
                                        rpn_anchor_generator=anchor_generator)

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
        backbone_model = get_model(backbone_id, weights='DEFAULT')
        backbone_layers = list(backbone_model.children())[:-1] # NOTE: iterating over named modules didn't work
        backbone_model = torch.nn.Sequential(*backbone_layers)
        backbone_model.out_channels = backbone_out_channels
        
        return backbone_model

    return # No backbone substitution available