import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from collections import Counter

from utils import ECGDataset
from models.fusion_model import ECG_Fusion


BATCH = 64
EPOCHS = 7
LR = 0.001


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset = ECGDataset(
        "data/processed/X.npy",
        "data/processed/y.npy"
    )

    n = len(dataset)
    train_n = int(0.8 * n)

    train_ds, val_ds = random_split(dataset, [train_n, n-train_n])

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH)


    # Class weights
    counts = Counter(dataset.y)
    total = sum(counts.values())

    weights = [total / counts[i] for i in range(len(counts))]
    weights = torch.tensor(weights).float().to(device)


    model = ECG_Fusion(num_classes=len(counts)).to(device)

    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=LR)


    for epoch in range(EPOCHS):

        model.train()
        correct = 0
        total_s = 0


        for x, y in train_loader:

            x, y = x.to(device), y.to(device)

            out = model(x)

            loss = loss_fn(out, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            pred = out.argmax(1)

            correct += (pred == y).sum().item()
            total_s += y.size(0)


        train_acc = correct / total_s


        # Validation
        model.eval()

        val_c = 0
        val_t = 0

        with torch.no_grad():

            for x, y in val_loader:

                x, y = x.to(device), y.to(device)

                out = model(x)

                pred = out.argmax(1)

                val_c += (pred == y).sum().item()
                val_t += y.size(0)


        val_acc = val_c / val_t

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f}")


    torch.save(model.state_dict(), "models/fusion_model.pth")
    print("Fusion model saved.")



if __name__ == "__main__":
    main()
