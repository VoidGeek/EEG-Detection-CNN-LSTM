# EEG Detection using CNN-LSTM Hybrid Model

This repository contains a Python script for preprocessing EEG data for seizure detection using EDF files.

## Prerequisites

Ensure you have the following installed:

- Python 3.8 or higher
- Required Python packages (see [Installation](#installation))
- **Dataset**: Download the [Seizure Epilepsy CHB-MIT EEG Dataset (Pediatric)](https://www.kaggle.com/datasets/abhishekinnvonix/seizure-epilepcy-chb-mit-eeg-dataset-pediatric), extract it, and place it in the `data` folder inside the project directory.


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/EEG-Detection-CNN-LSTM.git
   cd EEG-Detection-CNN-LSTM/scripts
   ``` 

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ``` 
3. Move to scripts folder in terminal:
   ```bash
   cd scripts
   ``` 
## Usage

### Preprocess Data
   ```bash
   python preprocess.py
   ``` 

### Train Model
   ```bash
   python train.py
   ``` 
### Evaluate Model
   ```bash
   python evaluate.py
   ``` 

## Folder Structure

```
Project/
├──  data/                      # Raw EEG data files (not included in the repository)
│   ├── chb01/                  # Patient 1
│   ├── chb02/                  # Patient 2
│   └── ...                    
├── processed_data/             # Preprocessed features and labels (automatically generated)
│    ├── features.npy 
│    └── labels.npy              
├── model/                       # Saved trained models
│    └── model.h5  
├── scripts/
│    ├── preprocess.py/         # Data preprocessing script
│    ├── train.py/              # Training script
│    └── evaluate.py/           # Evaluation script  
└── requirements.txt            # Project requirements
```
