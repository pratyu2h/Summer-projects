
from dataclasses import dataclass

import numpy as np
import pandas as pd
from ta import momentum, trend, volatility
import torch
from torch.utils.data import Dataset


BASE_FEATURES = ["Open", "High", "Low", "Close", "Volume"]
INDICATOR_FEATURES = ["rsi", "macd", "macd_signal", "bb_high", "bb_low", "bb_mid", "ema_20"]
TIME_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
LAG_FEATURES = ["return_1", "return_5", "return_15", "volume_change"]

ALL_FEATURES = BASE_FEATURES + INDICATOR_FEATURES + TIME_FEATURES + LAG_FEATURES


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rsi"] = momentum.RSIIndicator(df["Close"], window=14).rsi()

    macd = trend.MACD(df["Close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    bb = volatility.BollingerBands(df["Close"], window=20)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()

    df["ema_20"] = trend.EMAIndicator(df["Close"], window=20).ema_indicator()
    return df


def add_time_features(df: pd.DataFrame, timestamp_col: str = "Timestamp") -> pd.DataFrame:
    df = df.copy()
    ts = df[timestamp_col]
    hour = ts.dt.hour
    dow = ts.dt.dayofweek
    
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return_1"] = df["Close"].pct_change(1)
    df["return_5"] = df["Close"].pct_change(5)
    df["return_15"] = df["Close"].pct_change(15)
    df["volume_change"] = df["Volume"].pct_change(1)
    return df


def build_feature_frame(df: pd.DataFrame, timestamp_col: str = "Timestamp",
                         timestamp_is_unix: bool = True) -> pd.DataFrame:
    df = df.copy()
    if timestamp_is_unix:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit="s")
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    df = add_technical_indicators(df)
    df = add_time_features(df, timestamp_col)
    df = add_lag_features(df)

    
    df["target_return"] = df["Close"].pct_change().shift(-1)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)
    return df


@dataclass
class WindowedData:
    X: np.ndarray          # (N, seq_len, num_features)
    y: np.ndarray          # (N,) next-step return
    close: np.ndarray      # (N,) Close price at the END of each window, for reconstructing price from predicted returns
    timestamps: pd.Series  # (N,) timestamp at the end of each window


def make_windows(df: pd.DataFrame, seq_len: int = 60, features=None) -> WindowedData:
    features = features or ALL_FEATURES
    feat_arr = df[features].values.astype(np.float32)
    target_arr = df["target_return"].values.astype(np.float32)
    close_arr = df["Close"].values.astype(np.float32)

    X, y, close, ts = [], [], [], []
    for i in range(seq_len, len(df)):
        X.append(feat_arr[i - seq_len:i])
        y.append(target_arr[i - 1])       # return realized right after the window ends
        close.append(close_arr[i - 1])
        ts.append(df["Timestamp"].iloc[i - 1])

    return WindowedData(
        X=np.array(X), y=np.array(y), close=np.array(close),
        timestamps=pd.Series(ts),
    )


def time_split(n: int, train_frac: float = 0.7, val_frac: float = 0.15):
    """Chronological split -- never shuffle time series data. Returns (train_idx, val_idx, test_idx) slices."""
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n)


class BTCWindowDataset(Dataset):
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


if __name__ == "__main__":
    
    rng = np.random.default_rng(0)
    n = 500
    start = pd.Timestamp("2024-01-01")
    timestamps_unix = (start + pd.to_timedelta(np.arange(n), unit="min")).astype("int64") // 10**9

    price = 40000 + np.cumsum(rng.normal(0, 20, n))
    df_raw = pd.DataFrame({
        "Timestamp": timestamps_unix,
        "Open": price + rng.normal(0, 5, n),
        "High": price + np.abs(rng.normal(10, 5, n)),
        "Low": price - np.abs(rng.normal(10, 5, n)),
        "Close": price,
        "Volume": np.abs(rng.normal(100, 20, n)),
    })

    feat_df = build_feature_frame(df_raw)
    assert set(INDICATOR_FEATURES + TIME_FEATURES + LAG_FEATURES).issubset(feat_df.columns)
    assert "target_return" in feat_df.columns
    assert feat_df.isna().sum().sum() == 0, "warmup/tail NaNs should be fully dropped"

    windows = make_windows(feat_df, seq_len=30)
    assert windows.X.shape[1:] == (30, len(ALL_FEATURES))
    assert len(windows.X) == len(windows.y) == len(windows.close)

    tr, va, te = time_split(len(windows.X))
    assert tr.stop < va.stop < te.stop or te.stop == len(windows.X)

    ds = BTCWindowDataset(windows.X[tr], windows.y[tr])
    x0, y0 = ds[0]
    assert x0.shape == (30, len(ALL_FEATURES))

    print(f"[ok] feature frame rows={len(feat_df)}, columns={len(ALL_FEATURES)}")
    print(f"[ok] windows X={windows.X.shape}, y={windows.y.shape}")
    print(f"[ok] split sizes train/val/test = {tr.stop}/{va.stop-tr.stop}/{te.stop-va.stop}")
