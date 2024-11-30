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