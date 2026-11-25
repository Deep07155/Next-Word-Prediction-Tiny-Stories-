"""Same LSTM class as the user's TinyStories Kaggle training notebook."""

import torch.nn as nn


class NextWordLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_size=128,
                 num_layers=2, dropout=0.2, pad_id=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.lstm(x)
        last = self.dropout(output[:, -1, :])
        return self.fc(last)
