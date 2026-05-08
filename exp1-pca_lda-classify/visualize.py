from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np

MPL_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".matplotlib-cache")
os.environ.setdefault("MPLCONFIGDIR", MPL_CACHE_DIR)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import DatasetSplit
from model import NumpyLDA, NumpyPCA


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
CLASS_COLORS = (
    "#b53a3a",
    "#d6a82e",
    "#2f7f9f",
    "#5f5aa2",
)


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

    lda = NumpyLDA(n_components=min(split.num_classes - 1, pca_components))
    lda.fit(split.x_train, split.y_train)
    z_lda = lda.transform(x_all)

    pca_path = os.path.join(output_dir, f"{split.dataset_name}_pca_2d.png")
    lda_suffix = "2d" if z_lda.shape[1] >= 2 else "1d"
    lda_path = os.path.join(output_dir, f"{split.dataset_name}_lda_{lda_suffix}.png")

    _plot_2d(z_pca[:, :2], y_all, split.class_names, "PCA 2D Projection", "PC1", "PC2", pca_path)
    if z_lda.shape[1] >= 2:
        _plot_2d(z_lda[:, :2], y_all, split.class_names, "LDA 2D Projection", "LD1", "LD2", lda_path)
    else:
        _plot_1d(z_lda.reshape(-1), y_all, split.class_names, "LDA 1D Projection", "LD1", lda_path)
    return [pca_path, lda_path]


def _class_slices(y: np.ndarray) -> List[Tuple[int, np.ndarray]]:
    return [(int(label), y == label) for label in sorted(np.unique(y))]


def _class_color(label: int) -> str:
    return CLASS_COLORS[label % len(CLASS_COLORS)]


def _plot_2d(
    z: np.ndarray,
    y: np.ndarray,
    class_names: Dict[int, str],
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: str,
) -> None:
    if z.shape[1] != 2:
        raise ValueError("二维可视化要求输入 shape 为 [n_samples, 2]")

    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    for label, mask in _class_slices(y):
        ax.scatter(
            z[mask, 0],
            z[mask, 1],
            s=14,
            alpha=0.65,
            c=_class_color(label),
            edgecolors="none",
            label=class_names[label],
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_1d(
    values: np.ndarray,
    y: np.ndarray,
    class_names: Dict[int, str],
    title: str,
    xlabel: str,
    output_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=160)
    for label, mask in _class_slices(y):
        class_values = values[mask]
        ax.hist(
            class_values,
            bins=48,
            density=True,
            alpha=0.58,
            color=_class_color(label),
            label=class_names[label],
        )
        ax.axvline(
            class_values.mean(),
            color=_class_color(label),
            linestyle="--",
            linewidth=1.5,
            alpha=0.9,
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
