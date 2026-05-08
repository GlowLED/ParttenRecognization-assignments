from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, Iterable, List

from data import DATA_DIR, DatasetSplit, label_counts, load_dataset
from model import ExperimentResult, run_experiment


DEFAULT_DATASETS = ("mnist", "fashion_mnist")
DEFAULT_BACKENDS = ("numpy", "mindspore", "sklearn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp3: 朴素贝叶斯 MNIST 分类实验")
    parser.add_argument("--dataset", choices=("all", "mnist", "fashion_mnist"), default="all")
    parser.add_argument("--backend", choices=("all", "numpy", "mindspore", "sklearn"), default="all")
    parser.add_argument("--alpha", type=float, default=1.0, help="Laplace平滑系数")
    parser.add_argument("--binarize-threshold", type=float, default=0.5, help="二值化阈值")
    parser.add_argument("--no-binarize", action="store_true", help="不进行二值化")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--output-dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
    return parser.parse_args()


def _selected(value: str, defaults: Iterable[str]) -> List[str]:
    return list(defaults) if value == "all" else [value]


def _validate_args(args: argparse.Namespace) -> None:
    if args.alpha < 0:
        raise ValueError("--alpha 必须非负")
    if not args.no_binarize and not (0.0 <= args.binarize_threshold <= 1.0):
        raise ValueError("--binarize-threshold 必须在 [0, 1] 范围内")


def print_results(results: List[ExperimentResult]) -> None:
    print("\n实验结果")
    print("-" * 110)
    print(
        f"{'Dataset':<15} {'Backend':<12} {'Train Acc':>10} {'Test Acc':>10} "
        f"{'Train Time':>12} {'Predict Time':>13} {'Alpha':>8} {'Threshold':>10}"
    )
    print("-" * 110)
    for result in results:
        print(
            f"{result.dataset:<15} "
            f"{result.backend:<12} "
            f"{result.train_accuracy:>10.4f} "
            f"{result.test_accuracy:>10.4f} "
            f"{result.train_time_sec:>12.4f} "
            f"{result.predict_time_sec:>13.4f} "
            f"{result.alpha:>8.2f} "
            f"{result.binarize_threshold:>10.2f}"
        )
    print("-" * 110)


def write_summary_csv(results: List[ExperimentResult], output_dir: str) -> str:
    path = os.path.join(output_dir, "summary_metrics.csv")
    fieldnames = list(results[0].to_dict().keys()) if results else []
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_dict())
    return path


def write_predictions_csv(
    dataset_name: str,
    backend: str,
    predictions: np.ndarray,
    y_test: np.ndarray,
    class_names: Dict[int, str],
    output_dir: str,
) -> str:
    path = os.path.join(output_dir, f"{dataset_name}_{backend}_predictions.csv")
    fieldnames = ("sample_id", "true_label", "true_name", "pred_label", "pred_name", "correct")

    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(len(predictions)):
            true_label = int(y_test[idx])
            pred_label = int(predictions[idx])
            row = {
                "sample_id": idx,
                "true_label": true_label,
                "true_name": class_names[true_label],
                "pred_label": pred_label,
                "pred_name": class_names[pred_label],
                "correct": int(true_label == pred_label),
            }
            writer.writerow(row)
    return path


def write_results_json(results: List[ExperimentResult], output_dir: str) -> str:
    path = os.path.join(output_dir, "experiment_results.json")
    payload = {
        "results": [result.to_dict() for result in results],
    }
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    return path


def run_dataset(
    split: DatasetSplit,
    backends: List[str],
    alpha: float,
    binarize_threshold: float,
    output_dir: str,
) -> List[ExperimentResult]:
    print(f"\n数据集: {split.dataset_name}")
    print(f"训练集: {split.x_train.shape}, {label_counts(split.y_train, split.class_names)}")
    print(f"测试集: {split.x_test.shape}, {label_counts(split.y_test, split.class_names)}")

    results: List[ExperimentResult] = []
    for backend in backends:
        print(f"\n运行: {split.dataset_name} + BernoulliNB({backend})")
        result, predictions = run_experiment(
            dataset_name=split.dataset_name,
            backend=backend,
            x_train=split.x_train,
            y_train=split.y_train,
            x_test=split.x_test,
            y_test=split.y_test,
            alpha=alpha,
            binarize_threshold=binarize_threshold,
        )
        results.append(result)
        print(
            f"完成: train_acc={result.train_accuracy:.4f}, "
            f"test_acc={result.test_accuracy:.4f}, "
            f"train_time={result.train_time_sec:.4f}s, "
            f"predict_time={result.predict_time_sec:.4f}s"
        )

        pred_path = write_predictions_csv(
            dataset_name=split.dataset_name,
            backend=backend,
            predictions=predictions,
            y_test=split.y_test,
            class_names=split.class_names,
            output_dir=output_dir,
        )
        print(f"预测结果已保存: {pred_path}")

    return results


def main() -> None:
    args = parse_args()
    _validate_args(args)

    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    datasets = _selected(args.dataset, DEFAULT_DATASETS)
    backends = _selected(args.backend, DEFAULT_BACKENDS)
    binarize = not args.no_binarize
    binarize_threshold = args.binarize_threshold if binarize else 0.0

    print("=" * 88)
    print("Exp3: 朴素贝叶斯 MNIST 分类实验")
    print("=" * 88)
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    print(f"数据集: {datasets}")
    print(f"Backend: {backends}")
    print(f"Alpha: {args.alpha}")
    print(f"二值化: {binarize}, 阈值: {binarize_threshold}")

    all_results: List[ExperimentResult] = []

    for dataset_name in datasets:
        split = load_dataset(
            name=dataset_name,
            data_dir=data_dir,
            binarize=binarize,
            threshold=binarize_threshold,
        )
        results = run_dataset(split, backends, args.alpha, binarize_threshold, output_dir)
        all_results.extend(results)

    print_results(all_results)

    summary_path = write_summary_csv(all_results, output_dir)
    json_path = write_results_json(all_results, output_dir)

    print("\n输出文件")
    print(f"指标汇总: {summary_path}")
    print(f"JSON 结果: {json_path}")
    print(f"预测结果: {output_dir}/*_predictions.csv")


if __name__ == "__main__":
    main()
