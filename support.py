import torch
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

	epoch_loss = train_loss / len(trainloader) # Estimate epoch loss
	return epoch_loss

def evaluate(model, optimizer, testloader, device):
	metric = MeanAveragePrecision() # Will define map metric here
	val_loss = 0 # Total loss for epoch
	with torch.no_grad(): # No updates in evaluation
		for images, boxes, labels in testloader:
			#### TRAIN PASS - to get loss ####
			model.train()
			images = [image.to(device) for image in images]
			targets = [{'boxes':boxes[i].to(device), 'labels':labels[i].to(device)} 
						for i in len(boxes)]
			losses = model(images, targets) # Model returns dictionary of losses!
			val_loss += sum(loss for loss in losses.values())
			# Potentially calculate other metrics (accuracy, etc.)

			#### VAL PASS - for evaluating and calculating metrics ####
			model.eval()
			val_outputs = model(images) # Returns predictions in eval mode
			metric.update(targets, val_outputs) # will update map

	eval_results = metric.compute() # Calculate metrics after completing all batches
	epoch_loss = val_loss / len(testloader) # Estimate epoch loss

	return epoch_loss, eval_results




