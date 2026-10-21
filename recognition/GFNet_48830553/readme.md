GFNET Project
Name: Yoshihito Sugiyama
ID: 48830553
Dependencies: torch, torchvision, torcheval, matplotlib

Files
dataset.py
Contains functions to preprocess, split, and load data.

modules.py
Contains classes of the models and helper functions that will be used in other scripts.

predict.py
Includes the testing pipeline to analyse models for performance in accuracy and speed.

train.py
Includes the training pipeline to train and save models.

Introduction
This report will answer the overconfidence dilemma of machine learning models in automated triage. Specifically, the main focus will be the classifciation of human conditions (Alzheimer's vs Cognitively Normal) based on MRI scans (ADNI dataset). To perform the classification, the GFNet transformer model will be chosen due to it's known higher accuracy as a transformer model and speed due to the elementwise operations in Fourier domain with the only limitation being the O(nlog(n)) transformation calculations. As a baseline comparison, the ResNet18 model is chosen due to it's overall performance. To answer the dilemma three metrics will be used: difference in probabilities of each class during classification, overall high performance, and fast inference time. If there is a significant difference between confident correct answers and unconfident wrong answers, a manual threshold can be put in place to filter correct answers and incorrect answers that doctors can analyse instead. 