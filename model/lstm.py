import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers, n_labels, dropout):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_size, n_labels)

    def forward(self, input_ids):
        x = self.embedding(input_ids)

        lstm_out, (h_n, c_n) = self.lstm(x)
        h_last = h_n[-1] 
        h_last = self.dropout(h_last)

        logits = self.fc(h_last) 
        return logits