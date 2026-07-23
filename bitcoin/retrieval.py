"""
Retrieval layer -- the "RA" in Retrieval-Augmented Forecasting.

Two things get retrieved for a given current market window:
  1. Numeric neighbor outcomes (what actually happened after similar
     historical windows) -> blended into the final numeric prediction
     alongside the base GRU/LSTM forecast (`retrieval.py` + `RAFBlendHead`
     in model.py... see combined usage in train.py).
  2. The same neighbors, formatted as natural-language context -> fed into
     the Gemini prompt in `gemini_reasoning.py`. This is a standard RAG
     pattern: retrieve relevant context, then condition generation on it --
     just applied to market history instead of documents.

Index is built ONLY from the training split's embeddings. Querying it
with a validation/test window and retrieving training-set neighbors is
safe (no leakage: the neighbors' outcomes were already known at training
time). Never index validation/test windows themselves.
"""
from dataclasses import dataclass

import faiss
import numpy as np
import pandas as pd
import torch


@dataclass
class RetrievedNeighbor:
    distance: float
    outcome_return: float
    timestamp: pd.Timestamp


class MarketRetriever:
    def __init__(self, embeddings: np.ndarray, outcomes: np.ndarray, timestamps: pd.Series):
        """
        embeddings: (N, D) float32, from model.encode() on TRAIN windows only
        outcomes: (N,) the actual next-step return realized after each window
        timestamps: (N,) timestamp at the end of each window (for display/debugging)
        """
        assert len(embeddings) == len(outcomes) == len(timestamps)
        self.embeddings = embeddings.astype(np.float32)
        self.outcomes = outcomes.astype(np.float32)
        self.timestamps = timestamps.reset_index(drop=True)

        self.dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(self.embeddings)

    def query(self, query_embedding: np.ndarray, k: int = 10) -> list[RetrievedNeighbor]:
        """query_embedding: (D,) or (1, D). Returns k nearest historical windows."""
        q = query_embedding.reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(q, k)
        return [
            RetrievedNeighbor(
                distance=float(distances[0][j]),
                outcome_return=float(self.outcomes[indices[0][j]]),
                timestamp=self.timestamps.iloc[indices[0][j]],
            )
            for j in range(k) if indices[0][j] != -1
        ]

    def query_batch_stats(self, query_embeddings: np.ndarray, k: int = 10):
        """
        Vectorized version for scoring many windows at once (used during
        training/eval of the blend head). Returns, per query:
          weighted_mean: similarity-weighted mean of neighbor outcomes
                          (closer neighbors get more weight -- softmax over -distance)
          std: std of neighbor outcomes (retrieval "disagreement" -- a
               useful signal on its own: high std = no clean historical analog)
        """
        distances, indices = self.index.search(query_embeddings.astype(np.float32), k)
        neighbor_outcomes = self.outcomes[indices]  # (B, k)

        weights = np.exp(-distances)
        weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-8)

        weighted_mean = (weights * neighbor_outcomes).sum(axis=1)
        std = neighbor_outcomes.std(axis=1)
        return weighted_mean.astype(np.float32), std.astype(np.float32)


def build_retriever(model, train_loader, timestamps: pd.Series, device) -> MarketRetriever:
    """Runs the trained encoder over the full training set (no shuffling --
    caller should pass a non-shuffled loader) to build the retrieval index."""
    model.eval()
    embeddings, outcomes = [], []
    with torch.no_grad():
        for X, y in train_loader:
            X = X.to(device)
            emb = model.encode(X).cpu().numpy()
            embeddings.append(emb)
            outcomes.append(y.numpy())
    embeddings = np.concatenate(embeddings)
    outcomes = np.concatenate(outcomes)
    assert len(embeddings) == len(timestamps), (
        f"embedding count ({len(embeddings)}) must match timestamps ({len(timestamps)}) -- "
        "pass shuffle=False for the loader used here"
    )
    return MarketRetriever(embeddings, outcomes, timestamps)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from model import BTCForecaster
    from torch.utils.data import DataLoader
    from data import BTCWindowDataset

    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    N, T, F = 200, 30, 20
    X = rng.normal(size=(N, T, F)).astype(np.float32)
    y = rng.normal(scale=0.01, size=N).astype(np.float32)
    ts = pd.Series(pd.date_range("2024-01-01", periods=N, freq="min"))

    model = BTCForecaster(num_features=F, hidden_size=16, num_layers=1)
    loader = DataLoader(BTCWindowDataset(X, y), batch_size=32, shuffle=False)

    retriever = build_retriever(model, loader, ts, device=torch.device("cpu"))
    assert retriever.embeddings.shape == (N, model.embedding_dim)

    query_emb = model.encode(torch.from_numpy(X[:1])).detach().numpy()[0]
    neighbors = retriever.query(query_emb, k=5)
    assert len(neighbors) == 5
    assert neighbors[0].distance <= neighbors[-1].distance  # sorted by distance

    query_batch = model.encode(torch.from_numpy(X[:8])).detach().numpy()
    w_mean, std = retriever.query_batch_stats(query_batch, k=5)
    assert w_mean.shape == (8,) and std.shape == (8,)

    print(f"[ok] retriever index size={retriever.index.ntotal}, embedding_dim={retriever.dim}")
    print(f"[ok] single query returned {len(neighbors)} neighbors, nearest distance={neighbors[0].distance:.4f}")
    print(f"[ok] batch stats: weighted_mean sample={w_mean[:3]}, std sample={std[:3]}")
