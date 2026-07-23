# Bitcoin RAF (Retrieval-Augmented Forecasting)

Upgrade of the original single-cell LSTM next-price predictor into a
Retrieval-Augmented Forecasting architecture: a neural forecaster whose
predictions are blended with outcomes retrieved from similar historical
market windows via FAISS, plus a Gemini reasoning layer that explains the
result in plain language.

**Not financial advice.** This is a portfolio ML project demonstrating a
retrieval-augmented architecture on time-series data. A model predicting
next-minute returns with lower RMSE than a naive baseline does not by
itself imply a profitable trading strategy once fees, slippage, and
execution latency are accounted for — worth stating explicitly if you
present this.

## What changed from v1

`bitcoin project.ipynb` (repo root) is the original: a single-cell LSTM
trained full-batch on 5 raw OHLCV features, predicting raw price. Two
real bugs there worth naming if asked:

1. The target column was `Open` (index 0) but predictions were later
   inverse-transformed with `scaler_close` (fit on `Close`, a different
   column with a different min/max) — the "price" plotted didn't match
   what was actually predicted.
2. A `DataLoader` was constructed but never used in the training loop;
   training ran full-batch (entire training set as one batch every
   epoch) instead of mini-batch.

| | v1 | v2 (this folder) |
|---|---|---|
| Target | Raw price (mismatched scaler) | Next-step **return** (stationary, correctly scaled) |
| Features | 5 raw OHLCV | 20: OHLCV + RSI/MACD/Bollinger/EMA + cyclical time features + lag returns |
| Cell type | LSTM only | GRU or LSTM, configurable |
| Training | Full-batch, no validation loop | Mini-batch `DataLoader`, validation loop, early stopping, gradient clipping |
| Loss | MSE | Huber (less sensitive to outlier moves) |
| Architecture | Single forecaster | Forecaster + FAISS retrieval + learned blend head (RAF) |
| Explainability | None | Gemini reasoning layer grounded in retrieved historical analogs (RAG pattern) |

## Architecture

```
Window of 60 timesteps x 20 features
   -> GRU/LSTM encoder -> embedding (final hidden state)
   -> [raw forecast head]        -> raw_pred (next-step return)
   -> [FAISS lookup on embedding] -> k nearest historical windows -> their realized outcomes
   -> [blend head: MLP(raw_pred, retrieval_weighted_mean, retrieval_std)] -> final prediction
   -> [Gemini]: raw_pred + final prediction + retrieved analogs -> plain-language note
```

The retrieval index is built **only** from the training split's
embeddings — querying it with a validation/test window and retrieving
training-set neighbors is safe (their outcomes were already known during
training); the index never contains validation/test windows themselves.

The blend head is a small learned MLP rather than a fixed hand-picked
weight between "trust the model" and "trust retrieval" — it can learn to
lean on retrieval when historical neighbors tightly agree (low std) and
lean on the raw forecast when they don't (high std, no clean analog).

## Files

- `data.py` — feature engineering (`ta`-based technical indicators,
  cyclical time features, lag returns) and windowing, with the
  target/scaler bug fixed and returns used instead of raw price.
- `model.py` — `BTCForecaster` (GRU/LSTM + deeper FC head, exposes
  `.encode()` for retrieval) and `RAFBlendHead`.
- `retrieval.py` — `MarketRetriever`: FAISS index over training-set
  embeddings, k-NN lookup with similarity-weighted stats.
- `gemini_reasoning.py` — builds a RAG-style prompt from retrieved
  analogs + model outputs, calls Gemini.
- `train.py` — three-stage training (base model -> retriever -> blend
  head) with proper mini-batching, gradient clipping, early stopping,
  and an `rmse()` comparison of raw-only vs. blended predictions.
- `bitcoin_raf_v2.ipynb` — **run this one.** End-to-end Colab notebook:
  downloads data via Kaggle API, builds features, trains, evaluates,
  calls Gemini.

## Running it

1. Open `bitcoin_raf_v2.ipynb` in Colab.
2. Have a Kaggle API token (`kaggle.json`) ready.
3. Have a Gemini API key ready for the last section — store it in Colab
   Secrets as `GEMINI_API_KEY`, don't paste it in a cell.
4. Run all cells top to bottom. The model is small enough to train on
   CPU in a few minutes; GPU just makes it faster.

## Local development

Every module has an `if __name__ == "__main__":` smoke test on synthetic
data, no download needed:

```bash
pip install torch pandas numpy ta faiss-cpu scikit-learn matplotlib
python data.py        # feature engineering + windowing on synthetic OHLCV
python model.py       # forecaster + blend head forward/backward
python retrieval.py   # FAISS index build + query
python train.py       # full pipeline: base model -> retriever -> blend head -> eval
```

## Notes / limitations

- Next-minute return prediction on crypto is genuinely hard and mostly
  noise; a small RMSE improvement over a naive baseline is a reasonable
  ML result, not a trading signal. Be precise about that distinction —
  it's also usually what an interviewer is checking for.
- The retrieval index only ever contains training-set windows to avoid
  lookahead leakage; if you extend the horizon or retrain, rebuild it
  from the (possibly new) training split, not the full dataset.
- Gemini reasons over retrieved historical outcomes and the model's own
  numbers, not raw market data it's independently analyzing — same
  distinction as the pneumonia project's clinical layer: it's a
  communication/reasoning layer on top of a decision the quantitative
  model already made, not a second independent opinion.
