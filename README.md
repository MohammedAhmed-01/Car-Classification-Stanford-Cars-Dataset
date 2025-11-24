# Car-Classification-Stanford-Cars-Dataset
A deep learning project focused on building an image classification model capable of identifying 196 car models using the Stanford Cars Dataset. This repository contains the full training pipeline, preprocessing scripts, model architecture, evaluation metrics, and inference tools for real-world image predictions.
📌 Car Classification – Stanford Cars Dataset

A deep learning project focused on building an image classification model capable of identifying 196 car models using the Stanford Cars Dataset. This repository contains the full training pipeline, preprocessing scripts, model architecture, evaluation metrics, and inference tools for real-world image predictions.

🚀 Project Overview

This project uses convolutional neural networks (CNNs) and transfer learning techniques to classify car images with high accuracy. The Stanford Cars Dataset includes 16,185 images of cars annotated with 196 fine-grained categories, making it a challenging and detailed classification problem.

🧠 Key Features

✔️ Full preprocessing pipeline (resizing, augmentation, normalization)

✔️ Transfer Learning with state-of-the-art architectures (ResNet, EfficientNet, etc.)

✔️ Training, validation, and testing scripts

✔️ Accuracy, loss graphs, and performance evaluation

✔️ Confusion matrix visualization for model insights

✔️ Inference script for predicting car models from custom images

✔️ Fully reproducible setup with clear environment requirements

📂 Dataset

The project uses the Stanford Cars Dataset, which contains:

16,185 images

196 classes

High-resolution labeled car images
Dataset link (official Stanford page): User can download externally

🛠️ Tech Stack

Python

TensorFlow / Keras or PyTorch (depending on your implementation)

NumPy, Pandas, Matplotlib, Seaborn

Scikit-learn for evaluation metrics

📊 Model Performance

Includes:

Training & validation accuracy

Loss curves

Confusion matrix

Classification report (precision, recall, F1-score)

▶️ How to Use

Install dependencies

Download and extract the dataset

Run the training notebook or script

Use the inference script to classify car images

📷 Inference Example

Upload an image → the model predicts the car's make & model from the 196 classes.

📁 Repository Structure
├── data/                 # Dataset (images + labels)

├── notebooks/            # Training & evaluation notebooks

├── src/
│   ├── dataset.py        # Preprocessing & augmentation

│   ├── model.py          # CNN / Transfer Learning model

│   ├── train.py          # Training script

│   ├── evaluate.py       # Model evaluation

│   └── predict.py        # Inference script

├── saved_models/         # Trained weights

├── results/              # Graphs, confusion matrix, logs

└── README.md             # Project documentation
