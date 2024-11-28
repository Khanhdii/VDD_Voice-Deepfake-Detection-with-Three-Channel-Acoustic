# Update the README content to include dataset and feature information.
updated_readme_content = """
# Voice Spoofing Detection and Feature Extraction

This repository contains two Jupyter Notebooks designed for feature extraction and voice spoofing detection using machine learning and deep learning techniques.

## File Descriptions

### 1. `exact-feature-teep.ipynb`
**Purpose**: This notebook focuses on extracting audio features for voice-related applications, such as identifying spoofed voices or analyzing audio signals.

- **Libraries Used**:
  - `numpy`, `pandas`: Data manipulation and analysis.
  - `librosa`: Audio signal processing.
  - `tensorflow.keras`: Deep learning framework.
  - `spafe.features.cqcc`: Extracting CQCC features from audio data.
  - `matplotlib`: Visualization.
  - `tqdm`: Progress bar for loops.

- **Main Steps**:
  1. Load and parse dataset using pre-defined protocols.
  2. Extract audio features, particularly CQCC.
  3. Combine train, dev, and evaluation datasets for further processing.

---

### 2. `teep-voice-spoofing-detection.ipynb`
**Purpose**: This notebook builds and evaluates deep learning models for detecting voice spoofing.

- **Libraries Used**:
  - `numpy`, `pandas`: Data manipulation and preprocessing.
  - `librosa`: Audio processing.
  - `tensorflow.keras`, `keras.optimizers.Adam`: Deep learning models and optimization.
  - `seaborn`, `matplotlib`: Data visualization.
  - `sklearn.metrics`: Evaluation metrics (e.g., ROC curves, confusion matrices).

- **Main Steps**:
  1. Prepare audio datasets by reading and preprocessing `.flac` files.
  2. Implement deep learning models (ResNet50, MobileNet, EfficientNetB0).
  3. Train models using `MirroredStrategy` for distributed training.
  4. Evaluate performance using accuracy, ROC curves, and confusion matrices.

---

## Dataset and Features

### Dataset
The audio dataset used in this project is **ASVSpoof2019**, which contains `.flac` files for voice spoofing detection. It follows a structured protocol for train, development, and evaluation sets.

- **Source**: [ASVSpoof2019 Dataset on Kaggle](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset)

### Feature Dataset
The extracted features for the **ASVSpoof2019** dataset are also provided for convenience. These features include MFCC, CQCC, Spectrogram representations.

- **Source**: [Feature Dataset for ASVSpoof2019 on Kaggle](https://www.kaggle.com/datasets/caophankhnhduy/feature-la-asvspoof2019/data)

---

## Setup Instructions

### Prerequisites
1. Python 3.7+
2. Install the required libraries by running:
   ```bash
   pip install -r requirements.txt
