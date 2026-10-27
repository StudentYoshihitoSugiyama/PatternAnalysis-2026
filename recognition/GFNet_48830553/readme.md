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
Flags:  

- --model ("RN18", "GFNET")
- --version ("BEST", "LATEST")

### train.py

Includes the training pipeline to train and save models.  
Flags:  

- --model ("RN18", "GFNET")

## Usage

### Requirements

- python=3.14.7
- torch=2.14.1
- torchvision=0.29.1
- torcheval=0.0.7
- matplotlib=3.11.2

### Instructions

After installing the dependencies within a python environment, run "train.py" with appropriate flags ("RN18" for ResNet18, "GFNET" for GFNet) to train either a ResNet18 model or a GFNet model. Run "predict.py" with appropriate flags ("RN18" for ResNet18, "GFNET" for GFNet, "BEST" for model with highest validation accuracy, "LATEST" for latest saved model)to test a ResNet18 model or a GFNet model to analyse performance.

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

#### Memory

Due to the memory restriction of a singular Nvidia A100, the max batch size had to be reduced.

#### Epoch

The initial max epoch was set to a small number (5) to determine feasibility, but was increased (150) once feasibility was validated.

#### Weight Decay

Weight decay is important to control overfitting; however, too much weight decay caused the model to not learn and was adjusted accordingly. Not all parameters such as positional embedding, bias, LayerNorm, and LayerScale required weight decay as those parameters were not capturing complex features and was removed from decay.

#### Droppaths

To improve regularisation and decrease overfitting, droppaths were introduced. Dropout and droppaths both improve regularisation, but dropout deactivates specific neurons while droppaths deactivates entire layer/blocks.

#### Data Augmentation

Data augmenation in the form of jitter, flips, affine transformation, erasing, and guassian blur are introduced to increase the effective training set with variation to reduce overfitting. With further investigation, random erasing is suspected to introduce too much variation and decreased performance of the model and was removed.

#### Learning rate

Learning rate was adjusted to a small number to allow smoother learning with smaller chance to overshoot minimas. Furthermore, warm up was introduced to allow the model to learn the appropriate directions of parameter changes.

#### Initial values

Initial values of parameters were adjusted for faster and stable convergence.

#### Patch size

Reduced patch size of the GFNet for finer and granular patching to improve local feature extraction

#### CUDA and Hardware

CUDA and Nvidia A100 related code are implemented to imporve training time by utilising the specific hardware capabilities

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
