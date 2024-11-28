
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

## Setup Instructions

### Prerequisites
1. Python 3.7+
2. Install the required libraries by running:
   ```bash
   pip install -r requirements.txt
   ```

### Required Libraries
- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `librosa`, `tqdm`
- `tensorflow`, `keras`
- `spafe` (for feature extraction)

### Running the Notebooks
1. Open Jupyter Notebook or JupyterLab.
2. Navigate to the respective `.ipynb` file.
3. Execute the cells sequentially to reproduce the results.

---

## Dataset
Both notebooks expect audio datasets in `.flac` format. The dataset should follow the protocol structure specified in the `ASVspoof2019` challenge. Update the file paths in the notebooks to match your local directory structure.

---

## Acknowledgments
These notebooks leverage state-of-the-art techniques for audio processing and spoofing detection. Special thanks to the developers of `librosa`, `tensorflow`, and `spafe` for providing robust tools for audio and deep learning research.

