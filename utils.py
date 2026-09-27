import numpy as np
import torch
from torch.utils.data import Dataset


class ECGDataset(Dataset):

    def __init__(self, X_path, y_path, label_map=None):

        self.X = np.load(X_path)
        self.y = np.load(y_path)

        if label_map is None:
            labels = sorted(set(self.y))
            self.label_map = {l: i for i, l in enumerate(labels)}
        else:
            self.label_map = label_map

        self.y = np.array([self.label_map[l] for l in self.y])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)

        return x, y
