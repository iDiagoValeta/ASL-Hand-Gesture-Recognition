# American Sign Language (ASL) Classifier

This project implements an American Sign Language (ASL) classifier using **MediaPipe** for hand landmark detection and a deep learning model (TensorFlow/Keras) for classification. It allows you to collect your own ASL alphabet dataset, train a classifier, and perform real-time sign language inference using your webcam.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
  - [1. Collect Images](#1-collect-images)
  - [2. Create Dataset](#2-create-dataset)
  - [3. Train the Deep Classifier](#3-train-the-deep-classifier)
  - [4. Real-time Inference](#4-real-time-inference)
- [Key Bindings in Inference](#key-bindings-in-inference)
- [Model Details](#model-details)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Dataset Collection**: Easily collect images for each ASL sign (A-Z and BLANK) using your webcam.
- **Hand Landmark Extraction**: Utilizes MediaPipe to extract 21 hand landmarks, normalizing their coordinates for robust classification.
- **Deep Learning Classifier**: Trains a deep neural network (TensorFlow/Keras) on the extracted hand landmarks.
- **Real-time Inference**: Performs live sign language recognition from your webcam, displaying the predicted sign and accumulating characters into a word.
- **Word Building Logic**: Implements a time-based mechanism to add characters to a word, preventing rapid, erroneous additions.
- **Word Clearing**: Clears the current word if a "BLANK" sign is held for a specified duration.

## Project Structure

- `data/`: Directory where collected images for each sign will be stored.
- `collect_imgs.py`: Script to capture images from your webcam for each ASL sign.
- `create_dataset.py`: Extracts and normalizes hand landmarks using MediaPipe, then saves the data.
- `train_deep_classifier.py`: Trains a neural network model and saves it along with the label encoder.
- `deep_inference_classifier.py`: Performs real-time classification using your webcam.
- `data.pickle`: (Generated) Contains processed landmark data and labels.
- `deep_model.h5`: (Generated) The trained Keras deep learning model.
- `label_encoder.pkl`: (Generated) LabelEncoder object to convert labels back to sign names.

## Setup and Installation

### 1. Create a Conda Environment (Recommended)

conda create -n asl_env python=3.9
conda activate asl_env

### 2. Install Dependencies

pip install -r requirements.txt

or 

pip install opencv-python mediapipe scikit-learn tensorflow keras numpy

# Usage

## 1. Collect Images

python collect_imgs.py

Press Q when you're ready to start capturing images for the current sign.
It captures 500 images per sign by default.
Hold your hand steady, ensure good lighting and a clear background.

## 2. Create Dataset

python create_dataset.py

Uses MediaPipe to detect and normalize hand landmarks.
If no hand is detected (for non-'BLANK' classes), a warning is printed.
For 'BLANK', it stores a series of zeros.

Data is saved as data.pickle.

## 3. Train the Deep Classifier

python train_deep_classifier.py (here you can choose between the normal and the deep classifier, both work correctly)

Loads data.pickle, splits into train/test sets.
Trains a deep neural network and prints accuracy.
Saves the model as deep_model.h5 and label encoder as label_encoder.pkl.

## 4. Real-time Inference

python deep_inference_classifier.py (you have to execute depending on the classifier that you have trained before)

Detects your hand, predicts ASL sign.
Displays predicted sign on screen.
Adds characters to a word only if held for 3 seconds.
Holding "BLANK" for 5 seconds clears the word.

Key Bindings in Inference
q: Quit the application.

c: Delete the last character.

spacebar: Add a space.

Model Details
The model is a Sequential Keras model with:

Two Dense layers using ReLU activation.

Dropout layers to reduce overfitting.

An output Dense layer with softmax activation.

Compiled with adam optimizer and categorical_crossentropy loss.

Contributing
Feel free to fork the repository, open issues, or submit pull requests.
