# Summer Projects

## Pneumonia / Multi-Disease Chest X-Ray Classifier

`pneumonia classifier.ipynb` (repo root) is the original binary
(Pneumonia vs Normal) ResNet-50 classifier. The **`pneumonia/`** folder
contains the upgraded version: multi-label classification across all 14
NIH ChestX-ray14 findings, focal loss for class imbalance, Grad-CAM
interpretability, and a Gemini-based clinical decision-support layer.
See `pneumonia/README.md` for architecture details and
`pneumonia/pneumonia_classifier_v2.ipynb` to run it (Colab, GPU required).

## Bitcoin RAF (Retrieval-Augmented Forecasting) Price Predictor

`bitcoin project.ipynb` (repo root) is the original 1-minute BTC/USD LSTM
price predictor. The **`bitcoin/`** folder contains the upgraded version:
technical indicators, a GRU/LSTM forecaster predicting returns instead of
raw price, FAISS retrieval over historical market windows, a learned
blend head, and a Gemini reasoning layer grounded in the retrieved
analogs (RAG pattern). See `bitcoin/README.md` for architecture details
and `bitcoin/bitcoin_raf_v2.ipynb` to run it (Colab).

To take it to the next level, the following is the future course of plan:
1) Replace LSTM with GRU (faster, sometimes better on smaller data) or Transformer
2) Add more dense layers, e.g. 128→64→32→1
3) Add RSI, MACD, Bollinger Bands, EMA using ta library
4) Add hour of day, day of week — BTC has daily patterns



---

## 1. Model Architecture

| Change | Options |
|--------|---------|
| **Cell type** | Replace LSTM with `GRU` (faster, sometimes better on smaller data) or `Transformer` |
| **Hidden size** | Try 64, 256, 512 |
| **Num layers** | Try 1, 2, 4 — deeper isn't always better |
| **Dropout** | 0.1 to 0.5 |
| **FC layers** | Add more dense layers, e.g. 128→64→32→1 |
| **Bidirectional LSTM** | `bidirectional=True` — sees sequence forwards and backwards |

---

## 2. Input Features

| Change | Options |
|--------|---------|
| **Technical indicators** | Add RSI, MACD, Bollinger Bands, EMA using `ta` library |
| **Time features** | Add hour of day, day of week — BTC has daily patterns |
| **Lag features** | Add returns: `Close.pct_change()` instead of raw price |
| **Volume signals** | Volume spikes often precede price moves |

---

## 3. Training Setup

| Change | Options |
|--------|---------|
| **Optimizer** | Try `AdamW`, `RMSprop` |
| **Learning rate** | Try 0.001, 0.0001 — or use a scheduler |
| **LR Scheduler** | `ReduceLROnPlateau` — reduces LR when loss stops improving |
| **Loss function** | Try `HuberLoss` — less sensitive to outliers than MSE |
| **Epochs** | More epochs + early stopping |
| **Batch size** | 32, 128, 256 |

---

## 4. Data Setup

| Change | Options |
|--------|---------|
| **Sequence length** | Try 30, 100, 200 — longer = more context |
| **Data size** | Use more than 25000 rows |
| **Timeframe** | Resample 1-min data to 5-min or 1-hour — less noise |
| **Train/test split** | Try 85/15 or walk-forward validation |
| **Target** | Predict returns (`pct_change`) instead of raw price — more stationary |

---

## 5. Regularization

| Change | Options |
|--------|---------|
| **Dropout** | Already have it, tune the value |
| **Weight decay** | Add `weight_decay=1e-5` to Adam |
| **Early stopping** | Stop training when val loss stops improving |
| **Gradient clipping** | `torch.nn.utils.clip_grad_norm_` — prevents exploding gradients in LSTMs |

---

## 6. Prediction Strategy

| Change | Options |
|--------|---------|
| **Multi-step output** | Predict next N steps directly instead of rolling |
| **Probabilistic output** | Predict mean + variance to get confidence intervals |
| **Ensemble** | Train multiple models, average predictions |

---


