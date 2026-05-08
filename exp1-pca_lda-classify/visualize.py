import os
from typing import List, Tuple

import numpy as np

MPL_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".matplotlib-cache")
os.environ.setdefault("MPLCONFIGDIR", MPL_CACHE_DIR)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import CLASS_NAMES, DatasetSplit
from model import NumpyLDA, NumpyPCA


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
CLASS_COLORS = {
    0: "#b53a3a",
    1: "#d6a82e",
}


def save_reduction_plots(
    split: DatasetSplit,
    output_dir: str = OUTPUT_DIR,
    pca_components: int = 2,
) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)

    x_all = np.concatenate([split.x_train, split.x_test], axis=0)
    y_all = np.concatenate([split.y_train, split.y_test], axis=0)

    pca = NumpyPCA(n_components=pca_components)
    pca.fit(split.x_train)
    z_pca = pca.transform(x_all)

    lda = NumpyLDA()
    lda.fit(split.x_train, split.y_train)
    z_lda = lda.transform(x_all)

    pca_path = os.path.join(output_dir, "pca_2d.png")
    lda_path = os.path.join(output_dir, "lda_1d.png")

    _plot_pca(z_pca, y_all, pca_path)
    _plot_lda(z_lda, y_all, lda_path)
    return [pca_path, lda_path]


def _class_slices(y: np.ndarray) -> List[Tuple[int, np.ndarray]]:
    return [(int(label), y == label) for label in sorted(np.unique(y))]


def _plot_pca(z: np.ndarray, y: np.ndarray, output_path: str) -> None:
    if z.shape[1] != 2:
        raise ValueError("PCA 可视化要求二维输入")

    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    for label, mask in _class_slices(y):
        ax.scatter(
            z[mask, 0],
            z[mask, 1],
            s=12,
            alpha=0.62,
            c=CLASS_COLORS[label],
            edgecolors="none",
            label=CLASS_NAMES[label],
        )

    ax.set_title("PCA 2D Projection")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_lda(z: np.ndarray, y: np.ndarray, output_path: str) -> None:
    values = z.reshape(-1)

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=160)
    for label, mask in _class_slices(y):
        class_values = values[mask]
        ax.hist(
            class_values,
            bins=48,
            density=True,
            alpha=0.58,
            color=CLASS_COLORS[label],
            label=CLASS_NAMES[label],
        )
        ax.axvline(
            class_values.mean(),
            color=CLASS_COLORS[label],
            linestyle="--",
            linewidth=1.5,
            alpha=0.9,
        )

    ax.set_title("LDA 1D Projection")
    ax.set_xlabel("LD1")
    ax.set_ylabel("Density")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
