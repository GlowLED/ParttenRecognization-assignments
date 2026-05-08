from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, Iterable, List

from data import DATA_DIR, DatasetSplit, label_counts, load_train_test_split
from model import ExperimentResult, PredictionBundle, run_experiment
from visualize import OUTPUT_DIR, save_reduction_plots


PCA_COMPONENTS = 2
DEFAULT_DATASETS = ("wine", "iris")
DEFAULT_BACKENDS = ("numpy", "mindspore_official")
DEFAULT_REDUCERS = ("raw", "pca", "lda")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp1: PCA/LDA + Logistic Regression 对比实验")
    parser.add_argument("--dataset", choices=("all", "wine", "iris"), default="all")
    parser.add_argument("--backend", choices=("all", "numpy", "mindspore_official"), default="all")
    parser.add_argument("--reducer", choices=("all", "raw", "pca", "lda"), default="all")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def _selected(value: str, defaults: Iterable[str]) -> List[str]:
    return list(defaults) if value == "all" else [value]


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_epochs <= 0:
        raise ValueError("--max-epochs 必须大于 0")
    if args.lr <= 0.0:
        raise ValueError("--lr 必须大于 0")
    if args.tol < 0.0:
        raise ValueError("--tol 不能小于 0")
    if args.patience <= 0:
        raise ValueError("--patience 必须大于 0")


def print_results(results: List[ExperimentResult]) -> None:
    print("\n实验结果")
    print("-" * 132)
    print(
        f"{'Dataset':<8} {'Backend':<20} {'Reducer':<8} {'Dims':>4} "
        f"{'Train Acc':>10} {'Test Acc':>10} {'Loss':>12} "
        f"{'Epochs':>8} {'Converged':>10} {'Time(s)':>9}"
    )
    print("-" * 132)
    for result in results:
        print(
            f"{result.dataset:<8} "
            f"{result.backend:<20} "
            f"{result.reducer:<8} "
            f"{result.n_components:>4d} "
            f"{result.train_accuracy:>10.4f} "
            f"{result.test_accuracy:>10.4f} "
            f"{result.final_loss:>12.6f} "
            f"{result.epochs_run:>8d} "
            f"{str(result.converged):>10} "
            f"{result.duration_sec:>9.3f}"
        )
    print("-" * 132)


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
    bundles: List[PredictionBundle],
    output_dir: str,
    class_names_by_dataset: Dict[str, Dict[int, str]],
) -> List[str]:
    all_path = os.path.join(output_dir, "test_predictions.csv")
    dataset_paths = {
        dataset: os.path.join(output_dir, f"{dataset}_test_predictions.csv")
        for dataset in sorted(class_names_by_dataset)
    }
    fieldnames = (
        "dataset",
        "backend",
        "reducer",
        "sample_id",
        "true_label",
        "true_name",
        "pred_label",
        "pred_name",
        "correct",
        "probabilities_json",
    )

    rows_by_dataset: Dict[str, List[Dict[str, object]]] = {dataset: [] for dataset in dataset_paths}
    all_rows: List[Dict[str, object]] = []
    for bundle in bundles:
        result = bundle.result
        class_names = class_names_by_dataset[result.dataset]
        for idx, sample_id in enumerate(bundle.sample_ids):
            true_label = int(bundle.true_labels[idx])
            pred_label = int(bundle.predicted_labels[idx])
            row = {
                "dataset": result.dataset,
                "backend": result.backend,
                "reducer": result.reducer,
                "sample_id": int(sample_id),
                "true_label": true_label,
                "true_name": class_names[true_label],
                "pred_label": pred_label,
                "pred_name": class_names[pred_label],
                "correct": int(true_label == pred_label),
                "probabilities_json": json.dumps(
                    {
                        class_names[class_id]: float(bundle.probabilities[idx, class_id])
                        for class_id in sorted(class_names)
                    },
                    ensure_ascii=False,
                ),
            }
            rows_by_dataset[result.dataset].append(row)
            all_rows.append(row)

    paths = [all_path]
    _write_rows(all_path, fieldnames, all_rows)
    for dataset, path in dataset_paths.items():
        _write_rows(path, fieldnames, rows_by_dataset[dataset])
        paths.append(path)
    return paths


def _write_rows(path: str, fieldnames: Iterable[str], rows: List[Dict[str, object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_results_json(results: List[ExperimentResult], output_dir: str, plot_paths: List[str]) -> str:
    path = os.path.join(output_dir, "experiment_results.json")
    payload = {
        "pca_components": PCA_COMPONENTS,
        "results": [result.to_dict() for result in results],
        "plots": plot_paths,
    }
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    return path


def run_dataset(
    split: DatasetSplit,
    backends: List[str],
    reducers: List[str],
    args: argparse.Namespace,
) -> List[PredictionBundle]:
    print("\n数据集")
    print(f"名称: {split.dataset_name}")
    print(f"训练集: {split.x_train.shape}, {label_counts(split.y_train, split.class_names)}")
    print(f"测试集: {split.x_test.shape}, {label_counts(split.y_test, split.class_names)}")

    bundles: List[PredictionBundle] = []
    for backend in backends:
        for reducer in reducers:
            print(f"\n运行路径: {split.dataset_name.upper()} + {backend.upper()} + {reducer.upper()} + Logistic Regression")
            bundle = run_experiment(
                dataset_name=split.dataset_name,
                backend=backend,
                reducer_name=reducer,
                x_train=split.x_train,
                y_train=split.y_train,
                x_test=split.x_test,
                y_test=split.y_test,
                test_ids=split.test_ids,
                num_classes=split.num_classes,
                max_epochs=args.max_epochs,
                lr=args.lr,
                tol=args.tol,
                patience=args.patience,
                seed=args.seed,
                pca_components=PCA_COMPONENTS,
            )
            result = bundle.result
            bundles.append(bundle)
            print(
                f"完成: dims={result.n_components}, "
                f"train_acc={result.train_accuracy:.4f}, "
                f"test_acc={result.test_accuracy:.4f}, "
                f"epochs={result.epochs_run}, "
                f"converged={result.converged}"
            )
    return bundles


def main() -> None:
    args = parse_args()
    _validate_args(args)

    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    datasets = _selected(args.dataset, DEFAULT_DATASETS)
    backends = _selected(args.backend, DEFAULT_BACKENDS)
    reducers = _selected(args.reducer, DEFAULT_REDUCERS)

    print("=" * 88)
    print("Exp1: 红白葡萄酒/Iris PCA-LDA 降维 + Logistic Regression 对比实验")
    print("=" * 88)
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    print(f"数据集: {datasets}")
    print(f"Backend: {backends}")
    print(f"Reducer: {reducers}")
    print(f"PCA 降维维度: {PCA_COMPONENTS}")
    print(f"训练参数: max_epochs={args.max_epochs}, lr={args.lr}, tol={args.tol}, patience={args.patience}")

    all_bundles: List[PredictionBundle] = []
    plot_paths: List[str] = []
    class_names_by_dataset: Dict[str, Dict[int, str]] = {}

    for dataset_name in datasets:
        split = load_train_test_split(
            dataset_name=dataset_name,
            data_dir=data_dir,
            test_size=args.test_size,
            seed=args.seed,
        )
        class_names_by_dataset[dataset_name] = split.class_names
        if not args.no_plots:
            print(f"\n生成 {dataset_name} 降维可视化图片...")
            saved = save_reduction_plots(split, output_dir=output_dir, pca_components=PCA_COMPONENTS)
            for path in saved:
                print(f"已保存: {path}")
            plot_paths.extend(saved)
        all_bundles.extend(run_dataset(split, backends, reducers, args))

    results = [bundle.result for bundle in all_bundles]
    print_results(results)

    summary_path = write_summary_csv(results, output_dir)
    prediction_paths = write_predictions_csv(all_bundles, output_dir, class_names_by_dataset)
    json_path = write_results_json(results, output_dir, plot_paths)

    print("\n输出文件")
    print(f"指标汇总: {summary_path}")
    for path in prediction_paths:
        print(f"测试集分类结果: {path}")
    print(f"JSON 结果: {json_path}")
    for path in plot_paths:
        print(f"可视化图片: {path}")


if __name__ == "__main__":
    main()
