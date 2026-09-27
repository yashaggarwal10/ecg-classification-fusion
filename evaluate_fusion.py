import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from utils import ECGDataset
from models.fusion_model import ECG_Fusion


BATCH_SIZE = 32


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load dataset
    dataset = ECGDataset(
        "data/processed/X.npy",
        "data/processed/y.npy"
    )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    num_classes = len(dataset.label_map)

    # Load fusion model
    model = ECG_Fusion(num_classes=num_classes).to(device)

    model.load_state_dict(
        torch.load("models/fusion_model.pth", map_location=device)
    )

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


    print("\n===== FUSION MODEL RESULTS =====\n")

    print("Classification Report:\n")

    print(classification_report(all_labels, all_preds))

    print("Confusion Matrix:\n")

    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()