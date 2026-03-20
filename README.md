# Topic Classification - SRIP Task (Chitrabasha)

## Objective
The goal of this project is to build a text classifier from scratch to classify text data into 24 predefined topics. The challenge involves handling a massive dataset of 10 million rows and keeping the model parameters strictly under 5 Billion without using any pre-trained models.

## Performance Overview
* **Final Accuracy:** 90.47%
* **Trainable Parameters:** 21,020,184
* **Dataset Size Utilized:** Trained on a representative sample of 100,000 rows from the main 10M Parquet file for hardware and memory optimization.

## Model Architecture Details
The model utilizes a custom Deep Neural Network built entirely from scratch using PyTorch:
* **Input Layer:** 20,000 features (synchronized exactly with TF-IDF Vectorization).
* **Hidden Layers:** 1024 neurons followed by 512 neurons.
* **Activations & Regularization:** Implements ReLU activation, Batch Normalization, and Dropout (0.3) to prevent overfitting.
* **Optimization:** Incorporates a Learning Rate Scheduler for training stability.

## Repository Structure
The project folder is structured to meet submission guidelines:
* `src/` : Directory containing all source code.
    * `model.py` : Contains the custom Deep Learning architecture (PyTorch).
    * `train_dl.py` : The main execution script to load data, vectorize, and train the model.
    * `utils.py` : Contains functions for text cleaning and data preprocessing.
* `model_weights.pth` : Saved state dictionary of the fully trained model achieving 90.47% accuracy.
* `report.pdf` : Detailed project documentation, methodology, and evaluation metrics.


# ###  Download Trained Model Weights
The model weights file is too large for GitHub (82MB). You can download it from the link below:
https://drive.google.com/file/d/1t9V6F3j2-BAkICitBv3qBm7I5Cef4VSZ/view?usp=sharing
## Setup and Execution Instructions

### 1. Prerequisites
Ensure you have Python installed on your system. Clone this repository to your local machine and install the required dependencies:
```bash
pip install pandas scikit-learn pyarrow fastparquet torch

Ensure the dataset file is placed in the appropriate directory as referenced in the code, then execute the following commands in your terminal:

Bash
cd src
python train_dl.py
Debugging & Hardware Optimization Notes
During the development phase, the following critical optimizations were implemented:

Shape Mismatch Resolution: Addressed RuntimeError: mat1 and mat2 shapes cannot be multiplied by strictly aligning the TF-IDF max_features parameter with the neural network's input_dim.

Environment Stabilization: Resolved OMP: Error #15 (OpenMP duplicate library initialization) to ensure smooth, uninterrupted training execution.

Hardware Setup: Model developed and trained locally utilizing a Lenovo LOQ system equipped with an RTX 5050 GPU.
