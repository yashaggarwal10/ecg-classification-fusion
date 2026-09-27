import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from utils import ECGDataset
from models.gru_model import ECG_GRU


BATCH_SIZE = 32


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset = ECGDataset(
        "data/processed/X.npy",
        "data/processed/y.npy"
    )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    num_classes = len(dataset.label_map)

    model = ECG_GRU(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("models/gru_model.pth", map_location=device))

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            out = model(x)

            preds = out.argmax(1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()
