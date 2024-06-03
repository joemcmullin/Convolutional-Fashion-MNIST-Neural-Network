
# Fashion MNIST Convolutional Neural Network

## Introduction to Convolutional Neural Networks

Convolutional Neural Networks (CNNs) are a class of deep learning algorithms that are particularly effective for analyzing visual data. Unlike traditional neural networks, CNNs use a specialized architecture that is designed to take advantage of the 2D structure of input data, such as images. 

### Key Features of CNNs:

- **Convolutional Layers**: These layers apply a series of filters (kernels) to the input image to create feature maps. Each filter detects different features such as edges, textures, and patterns.
- **Pooling Layers**: These layers reduce the spatial dimensions of the feature maps, which helps to reduce the number of parameters and computational cost. Pooling also helps in making the detection of features invariant to scale and orientation changes.
- **Fully Connected Layers**: After several convolutional and pooling layers, the network may include fully connected layers that are similar to those in traditional neural networks. These layers perform the final classification based on the extracted features.

### CNNs vs. Traditional Neural Networks:

- **Spatial Hierarchy**: CNNs maintain the spatial hierarchy of the image by using local connections and pooling, whereas traditional neural networks treat the input data as a flat vector.
- **Parameter Sharing**: CNNs use parameter sharing (same weights for different parts of the input) in convolutional layers, reducing the number of parameters and improving generalization.
- **Better Performance on Visual Data**: Due to their specialized architecture, CNNs significantly outperform traditional neural networks on tasks involving visual data, such as image classification, object detection, and image segmentation.

## Project Overview

This repository contains code to build, train, and visualize a Convolutional Neural Network (CNN) using the Fashion MNIST dataset. The project covers data loading, exploration, CNN model definition, training, evaluation, and visualization.

<img src="https://github.com/joemcmullin/Convolutional-Neural-Network/assets/3474363/ce068e60-1c29-49d1-907c-ee7abf861eca" alt="Description" width="50%"/>

### Viewing Combined Dataset Summary

Create and display a combined summary of the dataset to understand the distribution of classes in both training and test sets.

<img src="https://github.com/joemcmullin/Convolutional-MNIST-Fashon-Neural-Network/assets/3474363/918b6bf5-3c28-4bfc-a645-163995e0eac5" width="50%"/>

### CNN Model Definition

Define the architecture of the Convolutional Neural Network (CNN). The model includes multiple convolutional layers to extract features from images, pooling layers to reduce dimensionality, and dense layers for classification.

<img src="https://github.com/joemcmullin/Convolutional-MNIST-Fashon-Neural-Network/assets/3474363/31f09ba6-7154-40da-ab7a-2ddc55c7d0a1" alt="Description" width="50%"/>

### Plotting Training History

Visualize the training and validation metrics over the epochs to understand how the model is learning. This helps in identifying any overfitting or underfitting issues.

<img src="https://github.com/joemcmullin/Convolutional-MNIST-Fashon-Neural-Network/assets/3474363/14928280-c974-44fe-9e5a-6ed8e6f87d64" width="50%"/>

### Test prediction made on the test set

<img src="https://github.com/joemcmullin/Convolutional-MNIST-Fashon-Neural-Network/assets/3474363/e368ac67-0a39-498e-958c-beaa2755cbfc" width="50%"/>

### Confusion Matrix

Compute and visualize the confusion matrix to evaluate the model's performance. The confusion matrix provides insights into the types of errors the model is making.

<img src="https://github.com/joemcmullin/Convolutional-MNIST-Fashon-Neural-Network/assets/3474363/f02da942-b796-43c7-aad9-9d1da73fb6c1" width="50%"/>



## Requirements

Before running the code, ensure you have the following packages installed:

- TensorFlow
- Keras
- Matplotlib
- NumPy
- Pandas
- Scikit-learn
- Seaborn
- Graphviz
- IPython

You should install the above required packages before using the attached Juypter Notebook.

## Conclusion

This project demonstrates how to build, train, and visualize a Convolutional Neural Network using the Fashion MNIST dataset. The provided visualizations and animations help in understanding the training process and the model's performance. By following this project, you can gain insights into CNNs and their application in image classification tasks.

