import torch
import torch.nn.functional as F
from torch import nn


class CNN(nn.Module):
    """
    CNN text classifier using convolution + pooling.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        num_filters: int = 100,
        kernel_sizes: tuple = (3, 4, 5),
        dropout: float = 0.5,
    ):
        super().__init__()
        # Padding index is set to 0 to match the `<padding>` token in preprocessing.py
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0
        )

        # Multiple kernel sizes to capture different n-gram features
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embed_dim,
                    out_channels=num_filters,
                    kernel_size=k,
                )
                for k in kernel_sizes
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)

        # PyTorch Conv1d expects channels as the second dimension: (batch_size, channels, seq_len)
        embedded = embedded.permute(0, 2, 1)

        # Convolution -> ReLU -> Max pooling over time
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        pooled = [
            F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved
        ]

        # Concatenate features from the different kernel sizes
        cat = self.dropout(
            torch.cat(pooled, dim=1)
        )  # (batch_size, len(kernel_sizes) * num_filters)

        return self.fc(cat)


class LSTM(nn.Module):
    """
    BiLSTM classifier using a sequence encoder + pooling.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)

        # Multiply hidden dimension by 2 if bidirectional
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)

        # Pass embeddings through the LSTM
        output, (hidden, cell) = self.lstm(embedded)
        # output shape: (batch_size, seq_len, lstm_output_dim)

        # Sequence encoder + pooling: Max pooling across the sequence length
        pooled, _ = torch.max(output, dim=1)  # (batch_size, lstm_output_dim)

        pooled = self.dropout(pooled)
        return self.fc(pooled)
