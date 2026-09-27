# ECG Classification Using RNN, CNN, and Multimodal Fusion

Author: Yash Aggarwal  
Python Version: 3.11  
Framework: PyTorch (CPU)  

---

## 📌 Project Overview

This project implements an end-to-end ECG classification system using the PTB-XL dataset.

It includes:

- GRU-based RNN model for time-series learning
- 1D CNN model for feature extraction
- Multimodal fusion model combining RNN and CNN features
- Class imbalance handling using weighted loss
- Fully reproducible preprocessing pipeline

The system classifies ECG recordings into five diagnostic superclasses:

- NORM (Normal)
- MI (Myocardial Infarction)
- STTC (ST/T Changes)
- HYP (Hypertrophy)
- CD (Conduction Disturbance)

---

## 📂 Project Structure (Main Files)

ECG_CNN2/
│
├── data/
│ ├── ptbxl/
│ │ ├── records100/
│ │ └── ptbxl_database.csv
│ └── processed/
│ ├── X.npy
│ └── y.npy
│
├── models/
│ ├── gru_model.py
│ ├── cnn1d_model.py
│ └── fusion_model.py
│
├── preprocessing/
│ └── preprocess.py
│
├── utils.py
├── train.py
├── train_fusion.py
├── evaluate.py
├── evaluate_fusion.py
│
├── requirements.txt
└── README.md


---

## ⚙️ Environment Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
```

Activate:

Windows:
venv\Scripts\activate

Linux / Mac:
source venv/bin/activate

### 2. Install Dependencies

pip install -r requirements.txt

## 📊 Dataset Setup

### 1.Download PTB-XL from PhysioNet:
https://physionet.org/content/ptb-xl/

### 2.Extract into:
data/ptbxl/

### 3.Ensure the following exist:

data/ptbxl/records100/
data/ptbxl/ptbxl_database.csv

## 🔄 Data Preprocessing

Run preprocessing script:
```bash
python preprocessing/preprocess.py
```

This will generate:
```bash
data/processed/X.npy
data/processed/y.npy
```

## 🧠 Model Training
### Train RNN Model
```bash
python train.py
```

Model saved as:
```bash
models/gru_model.pth
```

### Train Fusion Model
```bash
python train_fusion.py
```

Model saved as:
```bash
models/fusion_model.pth
```

## 📈 Model Evaluation
### Evaluate RNN Model
```bash
python evaluate.py
```

### Evaluate Fusion Model
```bash
python evaluate_fusion.py
```

Outputs:
- Classification Report
- Confusion Matrix

## 🚀 Key Features

- Fully reproducible pipeline
- CPU-compatible training
- Class imbalance handling
- Modular architecture
- Multimodal learning
- Clean dataset processing

## ⚠️ Known Limitations

- CPU-only training limits model size
- Uses only 3 ECG leads
- No hyperparameter search
- No attention mechanism

## 🔮 Future Improvements

- Attention-based RNN
- 12-lead analysis
- Hyperparameter optimization
- Web-based deployment
- Image-based CNN integration

## 📚 References

- PTB-XL Dataset - PhysioNet
- PyTorch Documentation
- WFDB Toolbox

## 📩 Contact

Author: Yash Aggarwal