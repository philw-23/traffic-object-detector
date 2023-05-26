import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class TrafficDetectorDataset(Dataset):
    def __init__(self, image_frame, target_size, transforms):
        self.image_frame = image_frame # Contains all bounding boxes, may be more than one for each image
        self.images = self.image_frame['f_path'].unique() # List of unique images
        self.target_size = target_size # Target size for images
        self.transforms = transforms

    def __getitem__(self, index):
        image_path = self.images[index] # Get image of interest
        image = Image.open(image_path) # Open image
        original_size = image.size
        image = self.resize_image(image) # Resize image preserving aspect ration
    
        # Get applicable items in image
        image_boxes = self.image_frame.loc[self.image_frame['f_path'] == image_path]
        
        # Account for bounding boxes and bbox labels
        bboxes = [] # For storing bboxes
        labels = [] # For storing bbox labels
        for idx, row in image_boxes.iterrows():
            xmin = row['xmin']
            xmax = row['xmax']
            ymin = row['ymin']
            ymax = row['ymax']
            xmin, ymin, xmax, ymax = self.adjust_bounding_box(xmin, ymin, 
                                                              xmax, ymax, 
                                                              original_size) # Adjust bbox
    
            bboxes.append([xmin, ymin, xmax, ymax]) # Append new coordinates
            labels.append(row['label_id']) # Append label
            
        # Convert to tensors
        boxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Apply transforms
        if self.transforms:
            image, boxes = self.transforms(image, boxes)
        
        return image, boxes, labels
    
    def resize_image(self, image):
        resized_image = image.resize(self.target_size)

        return resized_image

    def adjust_bounding_box(self, xmin, ymin, xmax, ymax, original_im_size):
        width, height = original_im_size
        target_width, target_height = self.target_size

        # Calculate scale factors
        width_ratio = target_width / width
        height_ratio = target_height / height

        # Adjust bounding box coordinates
        xmin = xmin * width_ratio
        ymin = ymin * height_ratio
        xmax = xmax * width_ratio
        ymax = ymax * height_ratio
        
        return xmin, ymin, xmax, ymax
    
    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):

        images, bboxes, labels = zip(*batch) # Get all batch elements
        stacked_images = torch.stack([image for image in images]) # Stack images
        bboxes = [box for box in bboxes] # Note: Don't stack bboxes
        labels = [label for label in labels] # Note: Don't stack labels
        
        return stacked_images, bboxes, labels