import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from utils import ECGDataset
from models.gru_model import ECG_GRU


BATCH_SIZE = 128
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
    val_n = n - train_n

    train_ds, val_ds = random_split(dataset, [train_n, val_n])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    num_classes = len(dataset.label_map)

    model = ECG_GRU(num_classes=num_classes).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):

        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for x, y in train_loader:

            x, y = x.to(device), y.to(device)

            out = model(x)
            loss = loss_fn(out, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            train_loss += loss.item()

            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_acc = correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in val_loader:

                x, y = x.to(device), y.to(device)

                out = model(x)

                preds = out.argmax(1)

                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f}")

    torch.save(model.state_dict(), "models/gru_model.pth")
    print("Model saved.")


if __name__ == "__main__":
    main()
