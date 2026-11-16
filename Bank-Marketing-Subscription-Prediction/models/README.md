# Trained Models

This directory stores trained PyTorch model weights for the Bank Marketing Subscription Prediction project.

## Model: Weighted Logistic Regression

- Framework: PyTorch
- Architecture: Custom Logistic Regression
- Input features: 50
- Output units: 1
- Trainable parameters: 51
- Loss function: BCEWithLogitsLoss
- Optimizer: Stochastic Gradient Descent (SGD)
- Learning rate: 0.01
- Training epochs: 30
- Positive class weight: 7.5477
- Selected classification threshold: 0.63

## Model File

`weighted_model.pth`

The file contains the trained model parameters (state dictionary), not the entire model architecture.

The architecture must be defined in PyTorch before loading these weights.
