# GFNET Project

Name: Yoshihito Sugiyama
ID: 48830553
Dependencies: torch, torchvision, torcheval, matplotlib

## Files

### dataset.py

Contains functions to preprocess, split, and load data.

### modules.py

Contains classes of the models and helper functions that will be used in other scripts.

### predict.py

Includes the testing pipeline to analyse models for performance in accuracy and speed.

### train.py

Includes the training pipeline to train and save models.

## Introduction

### Problem

This report will focus on the overconfidence dilemma of machine learning models in automated triage and determine whether advance machine learning models are suitable for MRI classification. The classification will be based on human conditions between Alzheimer's (AD) and Cognitively Normal (NC) on the ADNI dataset.

### Model

The chosen baseline model is Residual Neural Network 18-Layers (ResNet18) and the chosen advanced model is Global Filter Network (GFNet). ResNet18 is chosen as a baseline model for it's overall performance in classification. GFNet is chosen due to it's fast O(nlog(n)) run time complexity with the accuraccy of a normal transformer model. GFNet has a run time O(nlog(n)) because the model replaces the self-attention layer with a global filter in the Fourier domain, which is an element-wise multiplicaiton, but with the only drawback being the O(nlog(n)) complexity of the Fourier Transformation and Inverse Transformation.

### Evaluation Process

To answer the dilemma, confidence of each winning class prediction will be analysed to determine whether there is a significant difference between confident right answers and unconfident wrong answers. Furthermore, to answer whether an advanced machine learning models are suitable for MRI classification, accuracy (0-1 loss) and infernce time will be measured. A machine learning model should have high accuracy to be useful for operational use to filter hard classification that can be handed to professionals instead. Moreover, a machine learning model should have fast inference speed to be comparable to professionals.

### Data Processing

The data is stored as NC classes and AD classes for both provided training and testing datasets. The training dataset is further splitted into validation (20%) and training (80%) data loaders. Furthermore, during the data split, the datasets are splitted with respect to the patients to ensure no data leakage occurs. Moreover, the data is splitted with even distribution of classes in the validation and training data loaders.

## Process

## Conclusion

## Feasibility Report

## References

## AI Reference
