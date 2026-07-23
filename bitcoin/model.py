import torch
import torch.nn as nn


class BTCForecaster(nn.Module):
    def __init__(self, num_features: int, hidden_size: int = 128, num_layers: int = 2,
                 cell_type: str = "gru", dropout: float = 0.2, bidirectional: bool = False):
        super().__init__()
        assert cell_type in ("gru", "lstm")
        rnn_cls = nn.GRU if cell_type == "gru" else nn.LSTM
        self.cell_type = cell_type
        self.rnn = rnn_cls(
            input_size=num_features, hidden_size=hidden_size, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0, batch_first=True, bidirectional=bidirectional,
        )
        rnn_out_dim = hidden_size * (2 if bidirectional else 1)
        self.embedding_dim = rnn_out_dim

        self.head = nn.Sequential(
            nn.Linear(rnn_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, seq_len, num_features) -> (B, embedding_dim), the final-timestep hidden state."""
        out, _ = self.rnn(x)
        return out[:, -1, :]  # last timestep's output, both directions concatenated if bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encode(x)
        return self.head(emb).squeeze(-1)  # (B,), raw predicted next-step return


class RAFBlendHead(nn.Module):

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, raw_pred: torch.Tensor, retrieval_mean: torch.Tensor, retrieval_std: torch.Tensor) -> torch.Tensor:
        stacked = torch.stack([raw_pred, retrieval_mean, retrieval_std], dim=1)
        return self.net(stacked).squeeze(-1)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, F = 4, 30, 20
    for cell_type in ("gru", "lstm"):
        for bidir in (False, True):
            model = BTCForecaster(num_features=F, hidden_size=32, num_layers=2,
                                   cell_type=cell_type, bidirectional=bidir)
            dummy = torch.randn(B, T, F)
            pred = model(dummy)
            emb = model.encode(dummy)
            assert pred.shape == (B,)
            assert emb.shape == (B, model.embedding_dim)
            loss = pred.pow(2).mean()
            loss.backward()
            print(f"[ok] cell_type={cell_type} bidirectional={bidir}: pred={tuple(pred.shape)} emb_dim={model.embedding_dim}")

    blend = RAFBlendHead(hidden=8)
    raw_pred = torch.randn(B)
    retrieval_mean = torch.randn(B)
    retrieval_std = torch.rand(B)
    blended = blend(raw_pred, retrieval_mean, retrieval_std)
    assert blended.shape == (B,)
    blended.sum().backward()
    print(f"[ok] RAFBlendHead forward+backward, output shape={tuple(blended.shape)}")
