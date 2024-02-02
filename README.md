# Fish Classification Data Science Project
## Overview
This repository contains code and documentation for the Fish Classification project, focusing on fish classification using a deep learning approach. The project involves training models to classify fish images into 9 different classes and different experiments.

## Files
All files contain proper documantation, overview:
### Classes
* CNN_Fish - model for fish classification
* FishDataset - dataset class to retrieve images
### Model Train and Testing
* train - train the model using K-Fold cross validation
* utils - functionf for reading and train/test splitting of data
### Pretrained Models
* model_architectures_main - main for using pretrained models as classifiers of images
* model_architectures_train_test - train,test,validate pretrained models
* model_architectures_utils.py - getters for different models and classifiers
### Classifiers with Feature Extractor models
* feature_extractor_main - main for using pretrained models as feature extractor

## Dataset
[Link to Kaggle Dataset](https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset)
### Structure
The dataset comprises 9 labels of fish:
### Original Images:
50 images for each fish (except one class with 30 images)
RGB images with varying sizes
![image](https://github.com/Qehbr/Fish-Classification-DS-Project/assets/49615282/7d289e91-3009-4ba5-b32a-e6b4caf7e265)


## Training and Evaluation
### Initial Model 
We initially used original images with k-fold Cross Validation (k=5). For each fold: 80/20 train-validation/test split. The best-performing fold achieved the following results:

![image](https://github.com/Qehbr/Fish-Classification-DS-Project/assets/49615282/8fb4a7d9-b71a-44b0-b3ea-961ea1dbd538)
![image](https://github.com/Qehbr/Fish-Classification-DS-Project/assets/49615282/f5a8fdec-1525-45ff-8eac-0c78524f89d9)
* Train Accuracy: 95.3%
* Validation Accuracy: 66.7%
* Test Accuracy: 54.7%

Examples of correctly/incorrectly/uncertain classified images:

![image](https://github.com/Qehbr/Fish-Classification-DS-Project/assets/49615282/c6977731-4984-410d-90ee-cc3b760e1090)
![image](https://github.com/Qehbr/Fish-Classification-DS-Project/assets/49615282/55bcef90-5d22-44ca-9757-b8258fcdb871)
![image](https://github.com/Qehbr/Fish-Classification-DS-Project/assets/49615282/0ab98c5b-e66a-447c-be99-580b1e34620f)


## Improvements
To enhance model performance, we implemented the following changes:

* Used augmented images (7 new roated images for each image)
* Hyperparameter tuning (learning rate, batch size, epochs)
* Data preprocessing (image normalization)
Results for the best fold after improvements:

![image](https://github.com/Qehbr/Fish-Classification-DS-Project/assets/49615282/b09bdaef-627d-4a1a-82f3-01047d44c2ed)
![image](https://github.com/Qehbr/Fish-Classification-DS-Project/assets/49615282/898f8954-d270-4b0d-bc79-3ed1fe8cdb93)
* Train Accuracy: 97.9%
* Validation Accuracy: 83.8%
* Test Accuracy: 84.9%

Examples of correctly/incorrectly/uncertain classified images:
![image](https://github.com/Qehbr/Fish-Classification-DS-Project/assets/49615282/9425b866-1ad3-426b-abdf-b1128d1a9130)
![image](https://github.com/Qehbr/Fish-Classification-DS-Project/assets/49615282/2873f86e-3d90-4c67-b85f-07e46a6e0af6)
![image](https://github.com/Qehbr/Fish-Classification-DS-Project/assets/49615282/078fbd66-118d-4f66-ab4c-7ee33116e3c6)



## Iterative Test-time Augmentation (ITA)
We experimented with ITA, achieving improved results but not surpassing the model without ITA:

![image](https://github.com/Qehbr/Fish-Classification-DS-Project/assets/49615282/742d43b6-a2d1-4729-95d5-529c81bbe141)
![image](https://github.com/Qehbr/Fish-Classification-DS-Project/assets/49615282/9d7f99f2-aa1d-4f28-96d3-e120602b93df)

* Train Accuracy: 99.9%
* Validation Accuracy: 81.2%
* Test Accuracy: 80.2%

Examples of correctly/incorrectly classified images:
![image](https://github.com/Qehbr/Fish-Classification-DS-Project/assets/49615282/ec790635-4760-440e-9942-99459ab5fee9)
![image](https://github.com/Qehbr/Fish-Classification-DS-Project/assets/49615282/c7583671-e405-451d-aca4-b03ac354d074)


## Class Addition 
A new class, "Janitor Fish," was added:

![image](https://github.com/Qehbr/Fish-Classification-DS-Project/assets/49615282/8dfe83c7-c89b-4e1d-a038-8cfc78566947)

Results were slightly worse but still good performance:

![image](https://github.com/Qehbr/Fish-Classification-DS-Project/assets/49615282/d8a7cb66-e7a9-4c42-bd6a-6a005bae3656)
![image](https://github.com/Qehbr/Fish-Classification-DS-Project/assets/49615282/ffdbde6e-9485-4a6f-9607-2d47db50e051)

* Train Accuracy: 93.5%
* Validation Accuracy: 80.5%
* Test Accuracy: 79.2%
  
## Model Comparison
We compared the performance of different deep learning models:

### VGG19:
* Validation Accuracy: 26.09%
* Test Accuracy: 26.74%
  
### ResNet18:
* Validation Accuracy: 98.55%
* Test Accuracy: 100%
  
### DenseNet121:
* Validation Accuracy: 98.55%
* Test Accuracy: 100%
  
### ResNet50:
* Validation Accuracy: 98.55%
* Test Accuracy: 100%
  
## Classical ML Algorithms:
We used ResNet18 as a feature extractor model and applied Random Forest, SVM, and KNN classifiers:

### Random Forest:
* Validation Accuracy: 97.1%
* Test Accuracy: 100%

### SVM:
* Validation Accuracy: 98.55%
* Test Accuracy: 100%

### KNN:
* Validation Accuracy: 98.55%
* Test Accuracy: 100%
  
## Fast Learning Experiment 
We conducted experiments to assess how quickly models learn the dataset. ResNet18 demonstrated the best performance in terms of fast learning.
![image](https://github.com/Qehbr/Fish-Classification-DS-Project/assets/49615282/e671e95b-9fe4-459a-a6f1-0aac1871a65c)



