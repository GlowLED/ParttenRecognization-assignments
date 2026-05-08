import os
import urllib.request
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RED_WINE_FILE = "winequality-red.csv"
WHITE_WINE_FILE = "winequality-white.csv"
UCI_BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality"

FEATURE_NAMES = (
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
)

CLASS_NAMES = {0: "red", 1: "white"}


@dataclass(frozen=True)
class WineDataset:
    x: np.ndarray
    y: np.ndarray
    feature_names: Tuple[str, ...] = FEATURE_NAMES
    class_names: Optional[Dict[int, str]] = None

    def __post_init__(self) -> None:
        if self.class_names is None:
            object.__setattr__(self, "class_names", CLASS_NAMES)


@dataclass(frozen=True)
class DatasetSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    feature_names: Tuple[str, ...] = FEATURE_NAMES
    class_names: Optional[Dict[int, str]] = None

    def __post_init__(self) -> None:
        if self.class_names is None:
            object.__setattr__(self, "class_names", CLASS_NAMES)


def _download_if_needed(filename: str, data_dir: str) -> str:
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        return filepath

    url = f"{UCI_BASE_URL}/{filename}"
    print(f"下载数据集: {url}")
    try:
        urllib.request.urlretrieve(url, filepath)
    except Exception as exc:
        raise RuntimeError(
            f"无法下载 {filename}。可手动从 {url} 下载并放到 {data_dir}"
        ) from exc
    return filepath


def _load_wine_csv(filepath: str, label: int) -> Tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(filepath, delimiter=";", skip_header=1, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != 12:
        raise ValueError(f"{filepath} 的格式异常，期望 12 列，实际形状为 {data.shape}")

    x = data[:, :11].astype(np.float32)
    y = np.full(x.shape[0], label, dtype=np.int64)
    return x, y


def load_wine_dataset(data_dir: str = DATA_DIR) -> WineDataset:
    """Load red/white wine data and build a wine-type binary dataset."""
    red_path = _download_if_needed(RED_WINE_FILE, data_dir)
    white_path = _download_if_needed(WHITE_WINE_FILE, data_dir)

    x_red, y_red = _load_wine_csv(red_path, label=0)
    x_white, y_white = _load_wine_csv(white_path, label=1)

    x = np.concatenate([x_red, x_white], axis=0)
    y = np.concatenate([y_red, y_white], axis=0)
    return WineDataset(x=x, y=y)


def standardize_train_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit standardization on train data only, then transform train/test."""
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (
        ((x_train - mean) / std).astype(np.float32),
        ((x_test - mean) / std).astype(np.float32),
        mean.astype(np.float32),
        std.astype(np.float32),
    )


def stratified_split(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size 必须是 (0, 1) 内的浮点数")

    rng = np.random.default_rng(seed)
    train_indices = []
    test_indices = []

    for label in np.unique(y):
        label_indices = np.flatnonzero(y == label)
        rng.shuffle(label_indices)

        n_test = max(1, int(round(label_indices.size * test_size)))
        test_indices.append(label_indices[:n_test])
        train_indices.append(label_indices[n_test:])

    train_idx = np.concatenate(train_indices)
    test_idx = np.concatenate(test_indices)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


def load_train_test_split(
    data_dir: str = DATA_DIR,
    test_size: float = 0.2,
    seed: int = 42,
) -> DatasetSplit:
    dataset = load_wine_dataset(data_dir)
    x_train, y_train, x_test, y_test = stratified_split(
        dataset.x,
        dataset.y,
        test_size=test_size,
        seed=seed,
    )
    x_train, x_test, mean, std = standardize_train_test(x_train, x_test)

    return DatasetSplit(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        mean=mean,
        std=std,
        feature_names=dataset.feature_names,
        class_names=dataset.class_names,
    )


def label_counts(y: np.ndarray) -> Dict[str, int]:
    return {CLASS_NAMES[int(label)]: int(np.sum(y == label)) for label in np.unique(y)}
