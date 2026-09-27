import torch
import torch.nn as nn


class ECG_GRU(nn.Module):

    def __init__(self, input_size=3, hidden_size=64, num_classes=4):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)

        return out

'''
🧠 Why GRU?

Fewer parameters than LSTM
Faster on CPU
Suitable for fixed-length sequences (1000)

Q1: Why use last hidden state?
Answer:
The final hidden state summarizes the temporal dynamics of the full sequence.

Q2: Why not LSTM?
Answer:
GRU is computationally lighter and sufficient for moderate-length sequences.

Q3: What does GRU learn?
Answer:
It captures long-term temporal dependencies in ECG waveforms.

'''