# TensorFlow Lite Object Detection Training and Conversion

## Table of Contents
1. Installing Nvidia Driver, Cuda and cuDNN
2. Installing Anaconda and TensorFlow GPU
3. Preparing our Workspace and Anaconda Virtual Environment Directory Structure
4. Gathering and Labeling our Dataset
5. Generating Training Data
6. Configuring the Training Pipeline
7. Training the Model
8. Exporting the Inference Graph

## 1. Installing Nvidia Driver, Cuda and cudNN
a. Download and install compatible Nvidia driver (600mb) with cuda driver (2.7gb)
b. Download cuDNN 8 for cuda 10 (800mb) and cuDNN 8 for cuda 11 (800mb)
c. Extract and copy cuDNNs inside cuda

## 2. Installing Anaconda and TensorFlow GPU
a. Install Anaconda
b. Open Anaconda terminal
c. create a virtual environment with following commands

```
conda create -n tensorflow pip python=3.8
```

Then activate the environment with

```
conda activate tensorflow
```
**Note that whenever you open a new Anaconda Terminal you will not be in the virtual environment. So if you open a new prompt make sure to use the command above to activate the virtual environment**


