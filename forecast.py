from collections import deque
import numpy as np


class ForecastModel:
    def __init__(self):
        self.history = deque(maxlen=30)

    def update(self, congestion_value):
        self.history.append(congestion_value)

    def predict_high_congestion(self, threshold=0.7):
        if not self.history:
            return False
        hist = np.array(self.history, dtype=float)
        norm = np.max(hist) if np.max(hist) > 0 else 1.0
        score = float(np.mean(hist / norm))
        return score >= threshold

    def predict_next(self):
        # Simple one-step forecast using exponential smoothing
        if not self.history:
            return 0.0, 0.0
        hist = np.array(self.history, dtype=float)
        alpha = 0.4
        forecast = hist[0]
        for x in hist[1:]:
            forecast = alpha * x + (1 - alpha) * forecast
        # Confidence inversely related to variance
        var = float(np.var(hist))
        conf = float(1.0 / (1.0 + var))
        return float(forecast), conf


class LSTMStub:
    def __init__(self, window: int = 12):
        self.window = int(window)
        self.history = deque(maxlen=self.window)

    def update(self, value: float) -> None:
        self.history.append(float(value))

    def forecast(self) -> tuple[float, float]:
        if not self.history:
            return 0.0, 0.0
        arr = np.array(self.history, dtype=float)
        # EMA as proxy
        alpha = 0.5
        y = arr[0]
        for x in arr[1:]:
            y = alpha * x + (1.0 - alpha) * y
        rmse = float(np.sqrt(np.mean((arr - np.mean(arr)) ** 2)))
        conf = float(max(0.0, 1.0 - rmse / (50.0)))
        return float(y), conf


class STGCNStub:
    def __init__(self, window: int = 12):
        self.window = int(window)
        self.history = deque(maxlen=self.window)

    def update(self, value: float) -> None:
        self.history.append(float(value))

    def forecast(self, neighbor_influence: float = 0.0) -> tuple[float, float]:
        if not self.history:
            return 0.0, 0.0
        arr = np.array(self.history, dtype=float)
        # Low-pass filter plus neighbor influence
        kernel = np.array([0.2, 0.3, 0.5])
        if len(arr) >= 3:
            y = float(np.convolve(arr, kernel, mode="valid")[-1])
        else:
            y = float(np.mean(arr))
        y = y * (1.0 + 0.1 * np.tanh(neighbor_influence))
        var = float(np.var(arr))
        conf = float(1.0 / (1.0 + var / 25.0))
        return float(y), conf


class EnsembleForecaster:
    def __init__(self):
        self.lstm = LSTMStub()
        self.stgcn = STGCNStub()

    def update(self, node_signal: float) -> None:
        self.lstm.update(node_signal)
        self.stgcn.update(node_signal)

    def predict(self, meta_features: dict | None = None) -> tuple[float, float, float, float, float]:
        # meta_features can include: growth, accident_flag, recent_errors, neighbor_influence
        meta_features = meta_features or {}
        neighbor_influence = float(meta_features.get("neighbor_influence", 0.0))
        growth = float(meta_features.get("cluster_growth", 0.0))
        accident = float(meta_features.get("accident", 0.0))

        q_lstm, c_lstm = self.lstm.forecast()
        q_st, c_st = self.stgcn.forecast(neighbor_influence=neighbor_influence)

        # gating alpha in [0,1]
        z = np.array([np.tanh(growth), accident, 1.0], dtype=float)
        w = np.array([1.2, -0.8, 0.2], dtype=float)
        alpha = float(1.0 / (1.0 + np.exp(-float(w @ z))))  # sigma(w^T z)
        q_ens = float(alpha * q_lstm + (1.0 - alpha) * q_st)
        c_ens = float(alpha * c_lstm + (1.0 - alpha) * c_st)
        return q_ens, c_ens, alpha, c_lstm, c_st


