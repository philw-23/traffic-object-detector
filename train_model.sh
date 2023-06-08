#!/bin/bash

# Load packages
echo "Installing required packages..."
pip install -r requirements.txt
echo "Package installations complete"

# Train model
echo "Running model training..."
python train.py
echo "Completed model training"
