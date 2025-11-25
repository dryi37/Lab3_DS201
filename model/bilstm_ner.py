import torch
import torch.nn as nn


class BiLSTM_NER(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers, n_labels, dropout):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)

        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_size * 2, n_labels)

    def forward(self, input_ids):
        emb = self.dropout(self.embedding(input_ids))
        lstm_out, _ = self.bilstm(emb)
        logits = self.fc(self.dropout(lstm_out))
        return logits
