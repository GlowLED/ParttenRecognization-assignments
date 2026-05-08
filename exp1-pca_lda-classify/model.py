from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def cross_entropy(probas: np.ndarray, labels: np.ndarray, eps: float = 1e-8) -> float:
    safe = np.clip(probas, eps, 1.0)
    return float(-np.mean(np.log(safe[np.arange(labels.shape[0]), labels.astype(np.int64)])))


def accuracy_from_proba(probas: np.ndarray, labels: np.ndarray) -> float:
    pred = np.argmax(probas, axis=1)
    return float(np.mean(pred == labels.reshape(-1)))


@dataclass(frozen=True)
class ExperimentResult:
    dataset: str
    backend: str
    reducer: str
    n_components: int
    train_accuracy: float
    test_accuracy: float
    final_loss: float
    epochs_run: int
    converged: bool
    duration_sec: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PredictionBundle:
    result: ExperimentResult
    sample_ids: np.ndarray
    true_labels: np.ndarray
    predicted_labels: np.ndarray
    probabilities: np.ndarray


def require_mindspore():
    try:
        import mindspore as ms
        import mindspore.dataset as ds
        import mindspore.nn as nn
        from mindspore import Tensor
        from mindspore.common import set_seed
        from mindspore.train import Model
        from mindspore.train.callback import Callback, LossMonitor, TimeMonitor
        from mindspore.train.metrics import Accuracy
    except ImportError as exc:
        raise RuntimeError("MindSpore 未安装，无法运行 MindSpore 官方风格路径") from exc
    return ms, ds, nn, Tensor, set_seed, Model, Callback, LossMonitor, TimeMonitor, Accuracy


class IdentityReducer:
    def __init__(self) -> None:
        self.components_: Optional[np.ndarray] = None

    @property
    def n_components(self) -> int:
        if self.components_ is None:
            raise RuntimeError("IdentityReducer 尚未 fit")
        return int(self.components_.shape[1])

    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> "IdentityReducer":
        del y
        self.components_ = np.eye(x.shape[1], dtype=np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            raise RuntimeError("IdentityReducer 尚未 fit")
        return np.asarray(x, dtype=np.float32)

    def fit_transform(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(x, y).transform(x)


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
    def __init__(self, n_components: Optional[int] = None, reg: float = 1e-4) -> None:
        self.target_components = n_components
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
        y = np.asarray(y, dtype=np.int64)
        labels = np.unique(y)
        if labels.size < 2:
            raise ValueError("LDA 至少需要两个类别")

        overall_mean = x.mean(axis=0)
        sw = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
        sb = np.zeros_like(sw)

        for label in labels:
            class_x = x[y == label]
            class_mean = class_x.mean(axis=0)
            centered = class_x - class_mean
            diff = (class_mean - overall_mean).reshape(-1, 1)
            sw += centered.T @ centered
            sb += class_x.shape[0] * (diff @ diff.T)

        sw += self.reg * np.eye(x.shape[1], dtype=np.float64)
        matrix = np.linalg.pinv(sw) @ sb
        eigvals, eigvecs = np.linalg.eig(matrix)
        order = np.argsort(np.real(eigvals))[::-1]
        max_components = min(labels.size - 1, x.shape[1])
        if self.target_components is None:
            n_components = max_components
        else:
            n_components = min(self.target_components, max_components)

        components = np.real(eigvecs[:, order[:n_components]]).astype(np.float32)
        norms = np.linalg.norm(components, axis=0, keepdims=True)
        components = components / np.where(norms < 1e-12, 1.0, norms)

        self.mean_ = overall_mean.reshape(1, -1).astype(np.float32)
        self.components_ = components
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("LDA 尚未 fit")
        return ((np.asarray(x, dtype=np.float32) - self.mean_) @ self.components_).astype(np.float32)

    def fit_transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(x, y).transform(x)


class NumpySoftmaxRegression:
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        lr: float = 0.1,
        max_epochs: int = 800,
        tol: float = 1e-7,
        patience: int = 20,
    ) -> None:
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.lr = lr
        self.max_epochs = max_epochs
        self.tol = tol
        self.patience = patience
        self.weights = np.zeros((input_dim, num_classes), dtype=np.float32)
        self.bias = np.zeros((1, num_classes), dtype=np.float32)
        self.losses: List[float] = []
        self.epochs_run = 0
        self.converged = False

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = np.asarray(x, dtype=np.float32) @ self.weights + self.bias
        return softmax(logits).astype(np.float32)

    def fit(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        y_onehot = np.eye(self.num_classes, dtype=np.float32)[y]
        stale_epochs = 0
        previous_loss: Optional[float] = None
        self.losses = []
        self.converged = False

        for epoch in range(1, self.max_epochs + 1):
            probas = self.predict_proba(x)
            loss = cross_entropy(probas, y)
            error = (probas - y_onehot) / x.shape[0]
            grad_w = x.T @ error
            grad_b = np.sum(error, axis=0, keepdims=True)

            self.weights -= self.lr * grad_w.astype(np.float32)
            self.bias -= self.lr * grad_b.astype(np.float32)

            self.losses.append(loss)
            self.epochs_run = epoch
            if previous_loss is not None and abs(previous_loss - loss) < self.tol:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    self.converged = True
                    break
            else:
                stale_epochs = 0
            previous_loss = loss

        return self.losses


class ConvergenceCallback:
    """Lightweight callback shaped like MindSpore official training callbacks."""

    def __init__(self, max_epochs: int, tol: float, patience: int) -> None:
        _, _, _, _, _, _, Callback, _, _, _ = require_mindspore()

        class _Inner(Callback):
            def __init__(self, outer: "ConvergenceCallback") -> None:
                super().__init__()
                self.outer = outer

            def on_train_epoch_end(self, run_context):
                cb_params = run_context.original_args()
                loss = _tensor_to_float(cb_params.net_outputs)
                self.outer.losses.append(loss)
                self.outer.epochs_run = int(cb_params.cur_epoch_num)
                if self.outer.previous_loss is not None and abs(self.outer.previous_loss - loss) < self.outer.tol:
                    self.outer.stale_epochs += 1
                    if self.outer.stale_epochs >= self.outer.patience:
                        self.outer.converged = True
                        run_context.request_stop()
                else:
                    self.outer.stale_epochs = 0
                self.outer.previous_loss = loss

            def on_train_end(self, run_context):
                cb_params = run_context.original_args()
                if self.outer.epochs_run == 0:
                    self.outer.epochs_run = min(self.outer.max_epochs, int(cb_params.cur_epoch_num))

        self.max_epochs = max_epochs
        self.tol = tol
        self.patience = patience
        self.losses: List[float] = []
        self.epochs_run = 0
        self.converged = False
        self.previous_loss: Optional[float] = None
        self.stale_epochs = 0
        self.callback = _Inner(self)


class MindSporeOfficialSoftmaxRegression:
    """Classifier adapted from the training pattern used in MindSpore/models."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        lr: float = 0.1,
        max_epochs: int = 800,
        tol: float = 1e-7,
        patience: int = 20,
        seed: int = 42,
        batch_size: Optional[int] = None,
    ) -> None:
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.lr = lr
        self.max_epochs = max_epochs
        self.tol = tol
        self.patience = patience
        self.seed = seed
        self.batch_size = batch_size
        self.net = None
        self.losses: List[float] = []
        self.epochs_run = 0
        self.converged = False

    def fit(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        ms, ds, nn, _, set_seed, Model, _, LossMonitor, TimeMonitor, Accuracy = require_mindspore()
        ms.set_context(mode=ms.PYNATIVE_MODE)
        set_seed(self.seed)

        class LinearClassifier(nn.Cell):
            def __init__(self, input_dim: int, num_classes: int) -> None:
                super().__init__()
                self.dense = nn.Dense(input_dim, num_classes, weight_init="zeros", bias_init="zeros")

            def construct(self, inputs):
                return self.dense(inputs)

        features = np.asarray(x, dtype=np.float32)
        labels = np.asarray(y, dtype=np.int32)
        batch_size = self.batch_size or features.shape[0]
        train_dataset = ds.NumpySlicesDataset(
            {"features": features, "labels": labels},
            shuffle=True,
        ).batch(batch_size)

        self.net = LinearClassifier(self.input_dim, self.num_classes)
        loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        optimizer = nn.SGD(self.net.trainable_params(), learning_rate=self.lr)
        model = Model(self.net, loss_fn, optimizer, metrics={"Accuracy": Accuracy()})

        convergence = ConvergenceCallback(self.max_epochs, self.tol, self.patience)
        del TimeMonitor
        callbacks = [
            LossMonitor(per_print_times=max(train_dataset.get_dataset_size() * self.max_epochs + 1, 1)),
            convergence.callback,
        ]
        model.train(self.max_epochs, train_dataset, callbacks=callbacks, dataset_sink_mode=False)
        self.losses = convergence.losses
        self.epochs_run = convergence.epochs_run or self.max_epochs
        self.converged = convergence.converged
        return self.losses

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("MindSpore 官方风格分类器尚未 fit")
        ms, _, _, Tensor, _, _, _, _, _, _ = require_mindspore()
        logits = self.net(Tensor(np.asarray(x, dtype=np.float32), ms.float32)).asnumpy()
        return softmax(logits).astype(np.float32)


def _tensor_to_float(value) -> float:
    if isinstance(value, (tuple, list)):
        value = value[0]
    if hasattr(value, "asnumpy"):
        value = value.asnumpy()
    return float(np.mean(value))


def build_reducer(name: str, num_classes: int, pca_components: int = 2):
    if name == "raw":
        return IdentityReducer()
    if name == "pca":
        return NumpyPCA(n_components=pca_components)
    if name == "lda":
        return NumpyLDA(n_components=min(num_classes - 1, pca_components))
    raise ValueError(f"未知降维方式: {name}")


def run_experiment(
    dataset_name: str,
    backend: str,
    reducer_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    test_ids: np.ndarray,
    num_classes: int,
    max_epochs: int,
    lr: float,
    tol: float,
    patience: int,
    seed: int,
    pca_components: int = 2,
) -> PredictionBundle:
    start = time.perf_counter()
    reducer = build_reducer(reducer_name, num_classes=num_classes, pca_components=pca_components)
    z_train = reducer.fit_transform(x_train, y_train)
    z_test = reducer.transform(x_test)

    if backend == "numpy":
        classifier = NumpySoftmaxRegression(
            input_dim=z_train.shape[1],
            num_classes=num_classes,
            lr=lr,
            max_epochs=max_epochs,
            tol=tol,
            patience=patience,
        )
        backend_label = "NumPy"
    elif backend == "mindspore_official":
        classifier = MindSporeOfficialSoftmaxRegression(
            input_dim=z_train.shape[1],
            num_classes=num_classes,
            lr=lr,
            max_epochs=max_epochs,
            tol=tol,
            patience=patience,
            seed=seed,
        )
        backend_label = "MindSporeOfficial"
    else:
        raise ValueError(f"未知 backend: {backend}")

    losses = classifier.fit(z_train, y_train)
    train_probas = classifier.predict_proba(z_train)
    test_probas = classifier.predict_proba(z_test)
    predicted = np.argmax(test_probas, axis=1).astype(np.int64)

    result = ExperimentResult(
        dataset=dataset_name,
        backend=backend_label,
        reducer=reducer_name.upper(),
        n_components=reducer.n_components,
        train_accuracy=accuracy_from_proba(train_probas, y_train),
        test_accuracy=accuracy_from_proba(test_probas, y_test),
        final_loss=losses[-1] if losses else cross_entropy(train_probas, y_train),
        epochs_run=int(getattr(classifier, "epochs_run", max_epochs)),
        converged=bool(getattr(classifier, "converged", False)),
        duration_sec=time.perf_counter() - start,
    )

    return PredictionBundle(
        result=result,
        sample_ids=np.asarray(test_ids, dtype=np.int64),
        true_labels=np.asarray(y_test, dtype=np.int64),
        predicted_labels=predicted,
        probabilities=test_probas,
    )
