from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

UCI_WINE_BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality"
RED_WINE_FILE = "winequality-red.csv"
WHITE_WINE_FILE = "winequality-white.csv"
WINE_FEATURE_NAMES = (
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
WINE_CLASS_NAMES = {0: "red", 1: "white"}

IRIS_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
IRIS_FILE = "iris.data"
IRIS_FEATURE_NAMES = (
    "sepal length",
    "sepal width",
    "petal length",
    "petal width",
)
IRIS_CLASS_NAMES = {
    0: "Iris-setosa",
    1: "Iris-versicolor",
    2: "Iris-virginica",
}


@dataclass(frozen=True)
class TabularDataset:
    name: str
    x: np.ndarray
    y: np.ndarray
    sample_ids: np.ndarray
    feature_names: Tuple[str, ...]
    class_names: Dict[int, str]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


@dataclass(frozen=True)
class DatasetSplit:
    dataset_name: str
    x_train: np.ndarray
    y_train: np.ndarray
    train_ids: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    test_ids: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    feature_names: Tuple[str, ...]
    class_names: Dict[int, str]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


def _download_if_needed(url: str, filename: str, data_dir: str) -> str:
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        return filepath

    print(f"下载数据集: {url}")
    try:
        urllib.request.urlretrieve(url, filepath)
    except Exception as exc:
        raise RuntimeError(f"无法下载 {filename}。可手动从 {url} 下载并放到 {data_dir}") from exc
    return filepath


def _load_wine_csv(filepath: str, label: int) -> Tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(filepath, delimiter=";", skip_header=1, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != 12:
        raise ValueError(f"{filepath} 的格式异常，期望 12 列，实际形状为 {data.shape}")

    x = data[:, :11].astype(np.float32)
    y = np.full(x.shape[0], label, dtype=np.int64)
    return x, y


def load_wine_dataset(data_dir: str = DATA_DIR) -> TabularDataset:
    red_path = _download_if_needed(
        f"{UCI_WINE_BASE_URL}/{RED_WINE_FILE}",
        RED_WINE_FILE,
        data_dir,
    )
    white_path = _download_if_needed(
        f"{UCI_WINE_BASE_URL}/{WHITE_WINE_FILE}",
        WHITE_WINE_FILE,
        data_dir,
    )

    x_red, y_red = _load_wine_csv(red_path, label=0)
    x_white, y_white = _load_wine_csv(white_path, label=1)
    x = np.concatenate([x_red, x_white], axis=0)
    y = np.concatenate([y_red, y_white], axis=0)
    sample_ids = np.arange(x.shape[0], dtype=np.int64)
    return TabularDataset(
        name="wine",
        x=x,
        y=y,
        sample_ids=sample_ids,
        feature_names=WINE_FEATURE_NAMES,
        class_names=WINE_CLASS_NAMES,
    )


def load_iris_dataset(data_dir: str = DATA_DIR) -> TabularDataset:
    iris_path = _download_if_needed(IRIS_URL, IRIS_FILE, data_dir)
    features = []
    labels = []
    label_to_id = {name: idx for idx, name in IRIS_CLASS_NAMES.items()}

    with open(iris_path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 5:
                raise ValueError(f"{iris_path} 中存在异常行: {line}")
            features.append([float(value) for value in parts[:4]])
            labels.append(label_to_id[parts[4]])

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    sample_ids = np.arange(x.shape[0], dtype=np.int64)
    return TabularDataset(
        name="iris",
        x=x,
        y=y,
        sample_ids=sample_ids,
        feature_names=IRIS_FEATURE_NAMES,
        class_names=IRIS_CLASS_NAMES,
    )


def load_dataset(name: str, data_dir: str = DATA_DIR) -> TabularDataset:
    if name == "wine":
        return load_wine_dataset(data_dir)
    if name == "iris":
        return load_iris_dataset(data_dir)
    raise ValueError(f"未知数据集: {name}")


def standardize_train_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (
        ((x_train - mean) / std).astype(np.float32),
        ((x_test - mean) / std).astype(np.float32),
        mean.astype(np.float32),
        std.astype(np.float32),
    )


def stratified_split_indices(
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
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
    return train_idx, test_idx


def split_dataset(
    dataset: TabularDataset,
    test_size: float = 0.2,
    seed: int = 42,
) -> DatasetSplit:
    train_idx, test_idx = stratified_split_indices(dataset.y, test_size=test_size, seed=seed)
    x_train_raw = dataset.x[train_idx]
    x_test_raw = dataset.x[test_idx]
    x_train, x_test, mean, std = standardize_train_test(x_train_raw, x_test_raw)

    return DatasetSplit(
        dataset_name=dataset.name,
        x_train=x_train,
        y_train=dataset.y[train_idx],
        train_ids=dataset.sample_ids[train_idx],
        x_test=x_test,
        y_test=dataset.y[test_idx],
        test_ids=dataset.sample_ids[test_idx],
        mean=mean,
        std=std,
        feature_names=dataset.feature_names,
        class_names=dataset.class_names,
    )


def load_train_test_split(
    dataset_name: str = "wine",
    data_dir: str = DATA_DIR,
    test_size: float = 0.2,
    seed: int = 42,
) -> DatasetSplit:
    dataset = load_dataset(dataset_name, data_dir)
    return split_dataset(dataset, test_size=test_size, seed=seed)


def label_counts(y: np.ndarray, class_names: Optional[Dict[int, str]] = None) -> Dict[str, int]:
    names = class_names or {int(label): str(label) for label in np.unique(y)}
    return {names[int(label)]: int(np.sum(y == label)) for label in np.unique(y)}
