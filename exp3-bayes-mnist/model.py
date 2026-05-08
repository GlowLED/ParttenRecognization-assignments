from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ExperimentResult:
    dataset: str
    backend: str
    train_accuracy: float
    test_accuracy: float
    train_time_sec: float
    predict_time_sec: float
    alpha: float
    binarize_threshold: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class NumpyBernoulliNB:
    def __init__(self, alpha: float = 1.0) -> None:
        if alpha < 0:
            raise ValueError("alpha 必须非负")
        self.alpha = alpha
        self.class_log_prior_: Optional[np.ndarray] = None
        self.feature_log_prob_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NumpyBernoulliNB":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        n_samples = X.shape[0]

        self.class_log_prior_ = np.zeros(n_classes, dtype=np.float64)
        self.feature_log_prob_ = np.zeros((n_classes, n_features), dtype=np.float64)

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            n_c = X_c.shape[0]

            self.class_log_prior_[idx] = np.log(n_c) - np.log(n_samples)

            feature_count = X_c.sum(axis=0)
            smoothed_count = feature_count + self.alpha
            smoothed_total = n_c + 2 * self.alpha

            self.feature_log_prob_[idx] = np.log(smoothed_count) - np.log(smoothed_total)

        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None or self.class_log_prior_ is None or self.feature_log_prob_ is None:
            raise RuntimeError("模型尚未 fit")

        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        log_proba = np.zeros((n_samples, n_classes), dtype=np.float64)

        for idx in range(n_classes):
            log_p1 = self.feature_log_prob_[idx]
            log_p0 = np.log(1.0 - np.exp(log_p1))

            log_proba[:, idx] = self.class_log_prior_[idx]
            log_proba[:, idx] += X @ log_p1 + (1.0 - X) @ log_p0

        return log_proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("模型尚未 fit")
        log_proba = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_proba, axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))


class MindSporeBernoulliNB:
    def __init__(self, alpha: float = 1.0) -> None:
        if alpha < 0:
            raise ValueError("alpha 必须非负")
        self.alpha = alpha
        self.class_log_prior_: Optional[np.ndarray] = None
        self.feature_log_prob_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None

    def _require_mindspore(self):
        try:
            import mindspore as ms
            import mindspore.ops as ops
            return ms, ops
        except ImportError as exc:
            raise RuntimeError("MindSpore 未安装") from exc

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MindSporeBernoulliNB":
        ms, ops = self._require_mindspore()

        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.int32)

        self.classes_ = np.unique(y_np)
        n_classes = len(self.classes_)
        n_features = X_np.shape[1]
        n_samples = X_np.shape[0]

        class_log_prior = np.zeros(n_classes, dtype=np.float32)
        feature_log_prob = np.zeros((n_classes, n_features), dtype=np.float32)

        for idx, c in enumerate(self.classes_):
            X_c = X_np[y_np == c]
            n_c = X_c.shape[0]

            class_log_prior[idx] = np.log(n_c) - np.log(n_samples)

            X_c_tensor = ms.Tensor(X_c, ms.float32)
            feature_count = ops.ReduceSum(keep_dims=False)(X_c_tensor, 0).asnumpy()

            smoothed_count = feature_count + self.alpha
            smoothed_total = n_c + 2 * self.alpha

            feature_log_prob[idx] = np.log(smoothed_count) - np.log(smoothed_total)

        self.class_log_prior_ = class_log_prior
        self.feature_log_prob_ = feature_log_prob
        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None or self.class_log_prior_ is None or self.feature_log_prob_ is None:
            raise RuntimeError("模型尚未 fit")

        ms, ops = self._require_mindspore()

        X_np = np.asarray(X, dtype=np.float32)
        X_tensor = ms.Tensor(X_np, ms.float32)

        n_samples = X_np.shape[0]
        n_classes = len(self.classes_)

        log_proba = np.zeros((n_samples, n_classes), dtype=np.float32)

        for idx in range(n_classes):
            log_p1 = self.feature_log_prob_[idx]
            log_p0 = np.log(1.0 - np.exp(log_p1))

            log_p1_tensor = ms.Tensor(log_p1.reshape(-1, 1), ms.float32)
            log_p0_tensor = ms.Tensor(log_p0.reshape(-1, 1), ms.float32)

            term1 = ops.MatMul(False, False)(X_tensor, log_p1_tensor)
            ones_like = ops.OnesLike()(X_tensor)
            term2 = ops.MatMul(False, False)(ones_like - X_tensor, log_p0_tensor)

            log_proba[:, idx] = self.class_log_prior_[idx] + term1.asnumpy().flatten() + term2.asnumpy().flatten()

        return log_proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("模型尚未 fit")
        log_proba = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_proba, axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))


class SklearnBernoulliNB:
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnBernoulliNB":
        from sklearn.naive_bayes import BernoulliNB

        self._model = BernoulliNB(alpha=self.alpha)
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("模型尚未 fit")
        return self._model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        if self._model is None:
            raise RuntimeError("模型尚未 fit")
        return float(self._model.score(X, y))


def create_model(backend: str, alpha: float = 1.0):
    if backend == "numpy":
        return NumpyBernoulliNB(alpha=alpha)
    if backend == "mindspore":
        return MindSporeBernoulliNB(alpha=alpha)
    if backend == "sklearn":
        return SklearnBernoulliNB(alpha=alpha)
    raise ValueError(f"未知 backend: {backend}")


def run_experiment(
    dataset_name: str,
    backend: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float = 1.0,
    binarize_threshold: float = 0.5,
) -> Tuple[ExperimentResult, np.ndarray]:
    model = create_model(backend, alpha)

    start = time.perf_counter()
    model.fit(x_train, y_train)
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    predictions = model.predict(x_test)
    predict_time = time.perf_counter() - start

    train_accuracy = model.score(x_train, y_train)
    test_accuracy = float(np.mean(predictions == y_test))

    result = ExperimentResult(
        dataset=dataset_name,
        backend=backend,
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy,
        train_time_sec=train_time,
        predict_time_sec=predict_time,
        alpha=alpha,
        binarize_threshold=binarize_threshold,
    )

    return result, predictions
