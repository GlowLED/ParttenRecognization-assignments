from __future__ import annotations

import gzip
import os
import struct
import urllib.request
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

MNIST_BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist"
FASHION_MNIST_BASE_URL = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion"

MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

MNIST_CLASS_NAMES = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4",
    5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
}

FASHION_MNIST_CLASS_NAMES = {
    0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
    5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot",
}


@dataclass(frozen=True)
class DatasetSplit:
    dataset_name: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    class_names: Dict[int, str]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def n_train(self) -> int:
        return self.x_train.shape[0]

    @property
    def n_test(self) -> int:
        return self.x_test.shape[0]

    @property
    def n_features(self) -> int:
        return self.x_train.shape[1]


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


def _read_idx_images(filepath: str) -> np.ndarray:
    with gzip.open(filepath, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"图像文件magic number错误: {magic} (期望2051)")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num_images, rows * cols).astype(np.float64)


def _read_idx_labels(filepath: str) -> np.ndarray:
    with gzip.open(filepath, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"标签文件magic number错误: {magic} (期望2049)")
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)


def _load_idx_dataset(
    base_url: str,
    files: Dict[str, str],
    dataset_name: str,
    data_dir: str,
    class_names: Dict[int, str],
    binarize: bool = True,
    threshold: float = 0.5,
) -> DatasetSplit:
    paths = {}
    for key, filename in files.items():
        url = f"{base_url}/{filename}"
        paths[key] = _download_if_needed(url, filename, data_dir)

    x_train = _read_idx_images(paths["train_images"])
    y_train = _read_idx_labels(paths["train_labels"])
    x_test = _read_idx_images(paths["test_images"])
    y_test = _read_idx_labels(paths["test_labels"])

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    if binarize:
        x_train = (x_train > threshold).astype(np.float64)
        x_test = (x_test > threshold).astype(np.float64)

    return DatasetSplit(
        dataset_name=dataset_name,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        class_names=class_names,
    )


def load_mnist(
    data_dir: str = DATA_DIR,
    binarize: bool = True,
    threshold: float = 0.5,
) -> DatasetSplit:
    mnist_dir = os.path.join(data_dir, "mnist")
    return _load_idx_dataset(
        base_url=MNIST_BASE_URL,
        files=MNIST_FILES,
        dataset_name="mnist",
        data_dir=mnist_dir,
        class_names=MNIST_CLASS_NAMES,
        binarize=binarize,
        threshold=threshold,
    )


def load_fashion_mnist(
    data_dir: str = DATA_DIR,
    binarize: bool = True,
    threshold: float = 0.5,
) -> DatasetSplit:
    fashion_dir = os.path.join(data_dir, "fashion_mnist")
    return _load_idx_dataset(
        base_url=FASHION_MNIST_BASE_URL,
        files=MNIST_FILES,
        dataset_name="fashion_mnist",
        data_dir=fashion_dir,
        class_names=FASHION_MNIST_CLASS_NAMES,
        binarize=binarize,
        threshold=threshold,
    )


def load_dataset(
    name: str,
    data_dir: str = DATA_DIR,
    binarize: bool = True,
    threshold: float = 0.5,
) -> DatasetSplit:
    if name == "mnist":
        return load_mnist(data_dir, binarize, threshold)
    if name == "fashion_mnist":
        return load_fashion_mnist(data_dir, binarize, threshold)
    raise ValueError(f"未知数据集: {name}")


def label_counts(y: np.ndarray, class_names: Optional[Dict[int, str]] = None) -> Dict[str, int]:
    names = class_names or {int(label): str(label) for label in np.unique(y)}
    return {names[int(label)]: int(np.sum(y == label)) for label in np.unique(y)}
