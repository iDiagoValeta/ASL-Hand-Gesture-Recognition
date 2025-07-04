# Real-Time Sign Language Alphabet Recognition 🤟

This project is a real-time American Sign Language (ASL) alphabet recognizer. It uses your computer's webcam to detect hand gestures, translates them into the corresponding letter of the alphabet, and allows you to form words and sentences.

The system is built with Python, using MediaPipe for hand landmark detection and offers two different machine learning models for classification:
1.  A `RandomForestClassifier` from Scikit-learn (a classic, fast approach).
2.  A deep neural network built with `TensorFlow/Keras` (a more powerful, deep learning approach).

## ✨ Features

-   **Real-Time Detection**: Recognizes 26 ASL alphabet signs and a 'BLANK' state (no hand) directly from your webcam feed.
-   **Word Building Logic**: Assembles recognized letters into words. A letter is added after being held steady for 3 seconds. The word is cleared if a 'BLANK' state is held for 5 seconds.
-   **Interactive Controls**:
    -   `q`: Quit the application.
    -   `c`: Clear the last character.
    -   `spacebar`: Add a space to the current word.
-   **Complete ML Pipeline**: Includes scripts for every step of the process:
    1.  Image Collection
    2.  Dataset Creation
    3.  Model Training
    4.  Real-Time Inference

## ⚙️ How It Works

The project follows a standard machine learning pipeline:

1.  **Data Collection (`collect_imgs.py`)**: A script to capture images of each ASL sign from your webcam. It creates a dataset of images organized into folders for each class (A, B, C, ..., Z, BLANK).
2.  **Dataset Creation (`create_dataset.py`)**: This script processes the collected images. It uses `MediaPipe` to detect the 21 hand landmarks for each hand in an image, normalizes their coordinates, and saves them into a single `data.pickle` file.
3.  **Model Training (`train_classifier.py` / `train_deep_classifier.py`)**: You can choose your model. The script loads the `data.pickle` file and trains a classifier to recognize the sign based on the landmark data.
    -   `train_classifier.py` trains a `RandomForestClassifier`.
    -   `train_deep_classifier.py` trains a `Keras` neural network.
4.  **Inference (`inference_classifier.py` / `deep_inference_classifier.py`)**: The final step. This script opens your webcam, captures frames in real-time, uses MediaPipe to extract landmarks, and feeds them into your trained model to get a prediction, which is then displayed on the screen.

## 🚀 Getting Started

Follow these instructions to set up the project and run it on your local machine.

### Prerequisites

You need to have **Anaconda** or **Miniconda** installed on your system to manage the environment and dependencies.

### 1. Create and Activate the Conda Environment

First, clone the repository. Then, open your Anaconda Prompt or terminal and run the following commands to create a dedicated environment. We recommend using Python 3.9 for broad compatibility.

```bash
# Create a new conda environment named 'sign_language' with Python 3.9
conda create --name sign_language python=3.9

# Activate the newly created environment
conda activate sign_language
```

### 2. Install Dependencies

With the environment activated, install all the required libraries using the provided `requirements.txt` file.

```bash
# Install all dependencies
pip install -r requirements.txt
```

## 📋 Usage - Step-by-Step

To run the project, you must follow these steps in order.

### Step 1: Collect Image Data

Run the `collect_imgs.py` script. It will prompt you to get ready to show the sign for each letter of the alphabet. Press `Q` when you are ready to start collecting images for that specific letter. The script will automatically capture 500 images for each sign.

```bash
python collect_imgs.py
```

### Step 2: Create the Dataset from Images

After collecting the images, you need to process them to extract the hand landmarks. This script will create the `data.pickle` file that the models will use for training.

```bash
python create_dataset.py
```

### Step 3: Train the Model

You have two options for training. Choose one.

**Option A: Train the RandomForest Model**

This is a faster, classic machine learning approach. It saves the trained model as `model.p`.

```bash
python train_classifier.py
```

**Option B: Train the Deep Learning Model**

This uses a neural network and may yield higher accuracy. It saves the model as `deep_model.h5` and the label encoder as `label_encoder.pkl`.

```bash
python train_deep_classifier.py
```

### Step 4: Run the Real-Time Recognition

Now you're ready to see the magic! Run the inference script that corresponds to the model you trained in the previous step.

**If you trained with Option A (RandomForest):**

```bash
python inference_classifier.py
```

**If you trained with Option B (Deep Learning):**

```bash
python deep_inference_classifier.py
```

Point your hand at the webcam, and the application will start recognizing the signs!

## 📂 Project Structure

```
.
├── data/                     # Directory for collected images (created automatically)
├── collect_imgs.py           # Step 1: Script to collect image samples for each sign.
├── create_dataset.py         # Step 2: Script to process images and create a landmark dataset.
├── train_classifier.py       # Step 3 (Option A): Trains a RandomForest model.
├── train_deep_classifier.py  # Step 3 (Option B): Trains a Keras/TensorFlow model.
├── inference_classifier.py   # Step 4 (Option A): Runs real-time inference with the RandomForest model.
├── deep_inference_classifier.py # Step 4 (Option B): Runs real-time inference with the Keras model.
├── requirements.txt          # List of Python dependencies.
└── README.md                 # This file.
```
