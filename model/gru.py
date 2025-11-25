import torch
import torch.nn as nn

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers, n_labels, dropout):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)

        self.gru = nn.GRU(
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

        gru_out, h_n = self.gru(x)         
        h_last = h_n[-1]                    
        h_last = self.dropout(h_last)

        logits = self.fc(h_last)
        return logits
