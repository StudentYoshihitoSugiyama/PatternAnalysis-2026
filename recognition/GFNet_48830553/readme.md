# GFNET Project

Created by Yoshihito Sugiyama  
ID: 48830553  

## Files

### dataset.py

Contains functions to preprocess, split, and load data.

### modules.py

Contains classes of the models and helper functions that will be used in other scripts.

### predict.py

Includes the testing pipeline to analyse models for performance in accuracy and speed.

### train.py

Includes the training pipeline to train and save models.

## Usage

### Requirements

- torch=2.14.1
- torchvision=0.29.1
- torcheval=0.0.7
- matplotlib=3.11.2

### Instructions

After installing the dependencies, run "train.py" with appropriate flags to train either a ResNet18 model or a GFNet model. Run "predict.py" to test a ResNet18 model or a GFNet model to analyse performance.

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

Firstly, data processing and loading functions are created in the "dataset.py" file. Secondly, the main training loop function is constructed in the "train.py" file. Thirdly, the main testing function is created in the "predict.py" file. All repeated functions are saved in the "modules.py" file.

### Problems

- Memory: Reduced batch size to reduce overall memory usage in each batch to fit Nvidia A100
- Epoch: Initial max epoch was small (5) to check for feasibility, but was increased once feasibility was checked
- Hours: After initial run, estimated training time was reduced in batch files
- Weight Decay: Weight decay prevented learning in some parameters and was changed to be selective on which parameters to have weight decay
- Drop Paths: Added drop paths to improve ResNet18
- Data Augmentation: Added data augmentation to improve accuracy
- Learning Rate: Adjusted learning rate to allow warming up for improved accuracy
- Initial Values: Adjusting initial values of parameters for faster and stable convergence
- Patch Size: Reduce patch size for finer and granular patching to improve local feature extraction

### Results

### Discussion

## Conclusion

## Feasibility Report

*Name: Yoshihito Sugiyama*
*ID: 48830553*

### User Need, Scope & Acceptance Criteria

#### User

Patients for quicker and more reliable cancer checkups; doctors to reduce workload and allow focus on more difficult analysis

#### Purpose

To demonstrate that a model can reliably determine cancer/normal organs with an appropriate level of confidence

#### Acceptance Criteria

- If there is a significant difference between correct confident answers and wrong uncertain answers, a threshold can be placed to direct uncertain cases to doctors.
- If there are significant and reliable answers, the model may prove to be useful (0-1 Error and PR).
- If the inference time is shorter than the average doctor’s time to determine anomalous brain structure.

### Model Choice & Course Concepts

- By using FFT, GFNet can have the accuracy of a transformer model with the speed of FFTs
- No self-attention layer (replaced by global filter)
- No cross-attention layer (Classification problem)

### Preliminary Feasibility Evidence

#### ResNet18 (20 Epochs max)

- Final Train Acc: 0.7298
- Final Val Acc: 0.7220
- Final Test Acc: 0.6326
- Training Time: 809.85 seconds (13.50 minutes)

#### GFNet (20 Epochs max)

- Final Train Acc: 0.8356
- Final Val Acc: 0.7310
- Final Test Acc: 0.6258
- Training Time: 841.66 seconds (14.03 minutes)

### Risks, Budget & Fallback

- Less than 1 hour of training on the whole dataset with 20 epochs
- A100 GPU with 40GB - 80GB of memory
- Access to EAIT Labs computers, AWS GPUs, and Rangpur GPUs

## References

## AI Reference

Gemini 3.8 Low, Medium, and High was used to create the code and suggest improvements on the training loop and models.
