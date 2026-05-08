from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


def _stable_sigmoid(x: np.ndarray) -> np.ndarray:
    positive = x >= 0
    negative = ~positive
    z = np.empty_like(x, dtype=np.float32)
    z[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[negative])
    z[negative] = exp_x / (1.0 + exp_x)
    return z


def binary_cross_entropy(probas: np.ndarray, labels: np.ndarray, eps: float = 1e-7) -> float:
    y = labels.reshape(-1, 1).astype(np.float32)
    p = np.clip(probas.reshape(-1, 1), eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def binary_accuracy(probas: np.ndarray, labels: np.ndarray) -> float:
    pred = (probas.reshape(-1) >= 0.5).astype(np.int64)
    return float(np.mean(pred == labels.reshape(-1)))


@dataclass(frozen=True)
class ExperimentResult:
    backend: str
    reducer: str
    n_components: int
    train_accuracy: float
    test_accuracy: float
    final_loss: float


def _require_mindspore():
    try:
        import mindspore as ms
        import mindspore.nn as nn
        import mindspore.ops as ops
        from mindspore import Tensor
    except ImportError as exc:
        raise RuntimeError("MindSpore 未安装，无法运行 MindSpore 路径") from exc
    return ms, nn, ops, Tensor


class NumpyPCA:
    def __init__(self, n_components: int = 2) -> None:
        if n_components <= 0:
            raise ValueError("n_components 必须大于 0")
        self.target_components = n_components
        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None

    @property
    def n_components(self) -> int:
        if self.components_ is None:
            raise RuntimeError("PCA 尚未 fit")
        return int(self.components_.shape[1])

    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> "NumpyPCA":
        del y
        x = np.asarray(x, dtype=np.float32)
        self.mean_ = x.mean(axis=0, keepdims=True)
        centered = x - self.mean_

        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        explained_variance = (singular_values**2) / max(x.shape[0] - 1, 1)
        total_variance = np.sum(explained_variance)
        if total_variance <= 0.0:
            raise ValueError("PCA 无法处理总方差为 0 的数据")

        ratios = explained_variance / total_variance
        n_components = min(self.target_components, x.shape[1])

        self.components_ = vt[:n_components].T.astype(np.float32)
        self.explained_variance_ratio_ = ratios[:n_components].astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCA 尚未 fit")
        return ((np.asarray(x, dtype=np.float32) - self.mean_) @ self.components_).astype(np.float32)

    def fit_transform(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(x, y).transform(x)


class NumpyLDA:
    def __init__(self, reg: float = 1e-4) -> None:
        self.reg = reg
        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None

    @property
    def n_components(self) -> int:
        if self.components_ is None:
            raise RuntimeError("LDA 尚未 fit")
        return int(self.components_.shape[1])

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NumpyLDA":
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y)
        labels = np.unique(y)
        if labels.size != 2:
            raise ValueError("该 LDA 实现仅支持二分类")

        x0 = x[y == labels[0]]
        x1 = x[y == labels[1]]
        mean0 = x0.mean(axis=0)
        mean1 = x1.mean(axis=0)
        centered0 = x0 - mean0
        centered1 = x1 - mean1

        sw = centered0.T @ centered0 + centered1.T @ centered1
        sw = sw + self.reg * np.eye(x.shape[1], dtype=np.float32)
        direction = np.linalg.pinv(sw) @ (mean1 - mean0)
        norm = np.linalg.norm(direction)
        if norm <= 0.0:
            raise ValueError("LDA 得到零方向向量")

        self.mean_ = x.mean(axis=0, keepdims=True)
        self.components_ = (direction / norm).reshape(-1, 1).astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("LDA 尚未 fit")
        return ((np.asarray(x, dtype=np.float32) - self.mean_) @ self.components_).astype(np.float32)

    def fit_transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(x, y).transform(x)


class NumpyLogisticRegression:
    def __init__(self, input_dim: int, lr: float = 0.1, epochs: int = 800) -> None:
        self.lr = lr
        self.epochs = epochs
        self.weights = np.zeros((input_dim, 1), dtype=np.float32)
        self.bias = np.zeros((1,), dtype=np.float32)
        self.losses: List[float] = []

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = np.asarray(x, dtype=np.float32) @ self.weights + self.bias
        return _stable_sigmoid(logits).reshape(-1)

    def fit(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        x = np.asarray(x, dtype=np.float32)
        y_col = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        n_samples = x.shape[0]

        self.losses = []
        for _ in range(self.epochs):
            logits = x @ self.weights + self.bias
            probas = _stable_sigmoid(logits)
            error = probas - y_col

            grad_w = (x.T @ error) / n_samples
            grad_b = np.mean(error, axis=0)

            self.weights -= self.lr * grad_w.astype(np.float32)
            self.bias -= self.lr * grad_b.astype(np.float32)

            self.losses.append(binary_cross_entropy(probas, y_col))
        return self.losses


class MindSporePCA:
    def __init__(self, n_components: int = 2) -> None:
        if n_components <= 0:
            raise ValueError("n_components 必须大于 0")
        self.target_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None

    @property
    def n_components(self) -> int:
        if self.components_ is None:
            raise RuntimeError("PCA 尚未 fit")
        return int(self.components_.shape[1])

    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> "MindSporePCA":
        del y
        ms, _, ops, Tensor = _require_mindspore()
        ms.set_context(mode=ms.PYNATIVE_MODE)

        x_tensor = Tensor(np.asarray(x, dtype=np.float32), ms.float32)
        self.mean_ = ops.mean(x_tensor, axis=0, keep_dims=True)
        centered = x_tensor - self.mean_

        singular_values, _, right_vectors = ops.svd(centered, full_matrices=False, compute_uv=True)
        values = singular_values.asnumpy()
        explained_variance = (values**2) / max(x.shape[0] - 1, 1)
        total_variance = np.sum(explained_variance)
        if total_variance <= 0.0:
            raise ValueError("PCA 无法处理总方差为 0 的数据")

        ratios = explained_variance / total_variance
        n_components = min(self.target_components, x.shape[1])

        self.components_ = right_vectors[:, :n_components]
        self.explained_variance_ratio_ = ratios[:n_components].astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCA 尚未 fit")
        ms, _, ops, Tensor = _require_mindspore()
        x_tensor = Tensor(np.asarray(x, dtype=np.float32), ms.float32)
        projected = ops.matmul(x_tensor - self.mean_, self.components_)
        return projected.asnumpy().astype(np.float32)

    def fit_transform(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(x, y).transform(x)


class MindSporeLDA:
    def __init__(self, reg: float = 1e-4) -> None:
        self.reg = reg
        self.mean_ = None
        self.components_ = None

    @property
    def n_components(self) -> int:
        if self.components_ is None:
            raise RuntimeError("LDA 尚未 fit")
        return int(self.components_.shape[1])

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MindSporeLDA":
        ms, _, ops, Tensor = _require_mindspore()
        ms.set_context(mode=ms.PYNATIVE_MODE)

        x_np = np.asarray(x, dtype=np.float32)
        y_np = np.asarray(y)
        labels = np.unique(y_np)
        if labels.size != 2:
            raise ValueError("该 LDA 实现仅支持二分类")

        x_tensor = Tensor(x_np, ms.float32)
        x0 = Tensor(x_np[y_np == labels[0]], ms.float32)
        x1 = Tensor(x_np[y_np == labels[1]], ms.float32)

        mean0 = ops.mean(x0, axis=0, keep_dims=True)
        mean1 = ops.mean(x1, axis=0, keep_dims=True)
        centered0 = x0 - mean0
        centered1 = x1 - mean1

        sw = ops.matmul(centered0.T, centered0) + ops.matmul(centered1.T, centered1)
        sw = sw + self.reg * ops.eye(x_np.shape[1], x_np.shape[1], ms.float32)
        direction = ops.matmul(ops.inverse(sw), (mean1 - mean0).T)
        norm = ops.sqrt(ops.sum(direction * direction))
        direction = direction / ops.maximum(norm, Tensor(1e-12, ms.float32))

        self.mean_ = ops.mean(x_tensor, axis=0, keep_dims=True)
        self.components_ = direction
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("LDA 尚未 fit")
        ms, _, ops, Tensor = _require_mindspore()
        x_tensor = Tensor(np.asarray(x, dtype=np.float32), ms.float32)
        projected = ops.matmul(x_tensor - self.mean_, self.components_)
        return projected.asnumpy().astype(np.float32)

    def fit_transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(x, y).transform(x)


class MindSporeLogisticRegression:
    def __init__(self, input_dim: int, lr: float = 0.1, epochs: int = 800, seed: int = 42) -> None:
        self.input_dim = input_dim
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.net = None
        self.losses: List[float] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        ms, nn, _, Tensor = _require_mindspore()
        ms.set_context(mode=ms.PYNATIVE_MODE)
        ms.set_seed(self.seed)

        class BinaryLogisticCell(nn.Cell):
            def __init__(self, input_dim: int) -> None:
                super().__init__()
                self.dense = nn.Dense(input_dim, 1, weight_init="zeros", bias_init="zeros")

            def construct(self, inputs):
                return self.dense(inputs)

        self.net = BinaryLogisticCell(self.input_dim)
        loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
        optimizer = nn.SGD(self.net.trainable_params(), learning_rate=self.lr)
        train_cell = nn.TrainOneStepCell(nn.WithLossCell(self.net, loss_fn), optimizer)
        train_cell.set_train()

        features = Tensor(np.asarray(x, dtype=np.float32), ms.float32)
        labels = Tensor(np.asarray(y, dtype=np.float32).reshape(-1, 1), ms.float32)

        self.losses = []
        for _ in range(self.epochs):
            loss = train_cell(features, labels)
            self.losses.append(float(loss.asnumpy()))
        return self.losses

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("MindSpore 逻辑回归尚未 fit")
        ms, _, ops, Tensor = _require_mindspore()
        features = Tensor(np.asarray(x, dtype=np.float32), ms.float32)
        logits = self.net(features)
        return ops.sigmoid(logits).asnumpy().reshape(-1).astype(np.float32)


def run_numpy_experiment(
    reducer_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    lr: float,
    pca_components: int,
) -> ExperimentResult:
    reducer = _build_numpy_reducer(reducer_name, pca_components)
    z_train = reducer.fit_transform(x_train, y_train)
    z_test = reducer.transform(x_test)

    classifier = NumpyLogisticRegression(input_dim=z_train.shape[1], lr=lr, epochs=epochs)
    losses = classifier.fit(z_train, y_train)
    train_probas = classifier.predict_proba(z_train)
    test_probas = classifier.predict_proba(z_test)

    return ExperimentResult(
        backend="NumPy",
        reducer=reducer_name.upper(),
        n_components=reducer.n_components,
        train_accuracy=binary_accuracy(train_probas, y_train),
        test_accuracy=binary_accuracy(test_probas, y_test),
        final_loss=losses[-1],
    )


def run_mindspore_experiment(
    reducer_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    lr: float,
    pca_components: int,
    seed: int,
) -> ExperimentResult:
    reducer = _build_mindspore_reducer(reducer_name, pca_components)
    z_train = reducer.fit_transform(x_train, y_train)
    z_test = reducer.transform(x_test)

    classifier = MindSporeLogisticRegression(
        input_dim=z_train.shape[1],
        lr=lr,
        epochs=epochs,
        seed=seed,
    )
    losses = classifier.fit(z_train, y_train)
    train_probas = classifier.predict_proba(z_train)
    test_probas = classifier.predict_proba(z_test)

    return ExperimentResult(
        backend="MindSpore",
        reducer=reducer_name.upper(),
        n_components=reducer.n_components,
        train_accuracy=binary_accuracy(train_probas, y_train),
        test_accuracy=binary_accuracy(test_probas, y_test),
        final_loss=losses[-1],
    )


def _build_numpy_reducer(name: str, pca_components: int):
    reducers: Dict[str, object] = {
        "pca": NumpyPCA(n_components=pca_components),
        "lda": NumpyLDA(),
    }
    return reducers[name]


def _build_mindspore_reducer(name: str, pca_components: int):
    reducers: Dict[str, object] = {
        "pca": MindSporePCA(n_components=pca_components),
        "lda": MindSporeLDA(),
    }
    return reducers[name]
