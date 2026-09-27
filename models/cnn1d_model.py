import torch
import torch.nn as nn


class ECG_CNN1D(nn.Module):

    def __init__(self, input_channels=3, num_classes=5):
        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv1d(input_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Linear(128, num_classes)


    def forward(self, x):

        # x: (B, 1000, 3) → (B, 3, 1000)
        x = x.permute(0, 2, 1)

        feat = self.conv(x)
        feat = feat.squeeze(-1)

        out = self.fc(feat)

        return out, feat   # Return features too



'''
Q1: Why kernel sizes 7, 5, 3?
Answer:
Larger kernels capture broader waveform structure; smaller kernels refine local details.

Q2: Why AdaptiveAvgPool1d?
Answer:
It reduces feature dimension to fixed size regardless of input length.

'''