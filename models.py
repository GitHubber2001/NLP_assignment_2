import torch
from torch import nn


class CNN(nn.Module):
    def __init__(
        self,
        dictionary_size: int,
        embedding_dimention: int = 64,
        padding_index: int = 0,
        kernel_sizes: tuple[int, ...] = (2, 3, 4, 5),
        convolution_output_size: int = 64,
        num_classes: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.word_embedding = nn.Embedding(
            dictionary_size, embedding_dimention, padding_index
        )
        self.embedding_dropout = nn.Dropout(dropout)
        self.kernels = nn.ModuleList(
            [
                nn.Conv1d(embedding_dimention, convolution_output_size, k)
                for k in kernel_sizes
            ]
        )
        self.output_dropout = nn.Dropout(dropout)
        self.linear_layer = nn.Linear(
            convolution_output_size * len(kernel_sizes), num_classes
        )

    def forward(self, x: torch.Tensor):
        embedding = self.word_embedding(x)
        dropout_embedding = self.embedding_dropout(embedding)
        transposed = dropout_embedding.transpose(1, 2)
        pooled_values = []
        for kernel in self.kernels:
            nonlinear_kernels = torch.relu(kernel(transposed))
            pooled = torch.max(nonlinear_kernels, dim=2).values
            pooled_values.append(pooled)
        prediction = torch.cat(pooled_values, 1)
        prediction = self.output_dropout(prediction)
        return self.linear_layer(prediction)


class LSTM(nn.Module):
    pass
