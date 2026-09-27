import os
import wfdb
import numpy as np
import pandas as pd
from tqdm import tqdm
import ast


DATA_PATH = "data/ptbxl"
META_FILE = "data/ptbxl/ptbxl_database.csv"

LEADS = [0, 1, 7]
WINDOW_SIZE = 1000


# PTB-XL superclass mapping
LABEL_MAP = {
    "NORM": "NORM",

    # Myocardial infarction
    "IMI": "MI",
    "AMI": "MI",
    "LMI": "MI",
    "PMI": "MI",
    "MI": "MI",

    # ST/T changes
    "STTC": "STTC",

    # Hypertrophy
    "LVH": "HYP",
    "RVH": "HYP",
    "HYP": "HYP",

    # Conduction disturbances
    "LBBB": "CD",
    "RBBB": "CD",
    "LPFB": "CD",
    "CLBBB": "CD",
    "CRBBB": "CD",
    "CD": "CD"
}


def normalize(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


def load_metadata():
    return pd.read_csv(META_FILE)


def load_signal(record_name):
    path = os.path.join(DATA_PATH, record_name)
    record = wfdb.rdrecord(path)
    return record.p_signal


def segment_signal(signal):

    segments = []

    length = signal.shape[0]

    for i in range(0, length - WINDOW_SIZE + 1, WINDOW_SIZE):
        seg = signal[i:i + WINDOW_SIZE]
        segments.append(seg)

    return segments


def get_label(scp_dict):
    """
    Map detailed diagnosis to superclass
    """
    for key in scp_dict.keys():
        if key in LABEL_MAP:
            return LABEL_MAP[key]
    return None


def main():

    print("Loading metadata...")
    meta = load_metadata()

    X = []
    y = []

    print("Processing ECG signals...")

    for _, row in tqdm(meta.iterrows(), total=len(meta)):

        rec = row["filename_lr"]

        try:
            sig = load_signal(rec)

            sig = sig[:, LEADS]
            sig = normalize(sig)

            segments = segment_signal(sig)

            scp = ast.literal_eval(row["scp_codes"])

            label = get_label(scp)

            if label is None:
                continue

            for seg in segments:
                X.append(seg)
                y.append(label)

        except Exception:
            continue

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    print("Final dataset shape:", X.shape)
    print("Total labels:", len(set(y)))
    print("Label distribution:")
    print(pd.Series(y).value_counts())

    os.makedirs("data/processed", exist_ok=True)

    np.save("data/processed/X.npy", X)
    np.save("data/processed/y.npy", y)

    print("Saved processed dataset.")


if __name__ == "__main__":
    main()

'''
✅ What You Did Here
You built a fully reproducible ECG preprocessing pipeline from scratch.
Key steps:

Load PTB-XL metadata
Load ECG signals using WFDB
Select 3 leads: I, II, V2
Normalize signals (Z-score)
Segment signals (1000 samples)
Parse scp_codes
Map detailed diagnoses → 5 superclasses
Save as X.npy and y.npy


🧠 Why This Is Important

The original repo:
Used external pickle files
Had broken paths
Was not reproducible

Your pipeline:
Uses raw PTB-XL directly
Is OS-independent
Handles label mapping explicitly
Is fully reproducible

❓ What They May Ask
Q1: Why select only 3 leads?
Answer:
To reduce computational cost and simulate a minimal clinical setup while preserving informative spatial signals.

Q2: Why Z-score normalization?
Answer:
It standardizes amplitude variation across patients and improves model convergence.

Q3: Why map detailed scp codes to superclasses?
Answer:
PTB-XL contains granular diagnoses; mapping to superclasses reduces label sparsity and improves training stability.


'''
