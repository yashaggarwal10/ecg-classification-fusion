import torch
import torch.nn as nn

from models.gru_model import ECG_GRU
from models.cnn1d_model import ECG_CNN1D


class ECG_Fusion(nn.Module):

    def __init__(self, num_classes=5):
        super().__init__()

        self.rnn = ECG_GRU(hidden_size=64, num_classes=num_classes)
        self.cnn = ECG_CNN1D(num_classes=num_classes)

        self.fusion_fc = nn.Sequential(
            nn.Linear(64 + 128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )


    def forward(self, x):

        # RNN features
        rnn_out, _ = self.rnn.gru(x)
        rnn_feat = rnn_out[:, -1, :]

        # CNN features
        _, cnn_feat = self.cnn(x)

        # Fusion
        fused = torch.cat([rnn_feat, cnn_feat], dim=1)

        out = self.fusion_fc(fused)

        return out
