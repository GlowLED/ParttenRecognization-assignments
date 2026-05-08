import argparse
import os
from typing import List

from data import DATA_DIR, label_counts, load_train_test_split
from model import ExperimentResult, run_mindspore_experiment, run_numpy_experiment
from visualize import OUTPUT_DIR, save_reduction_plots


PCA_COMPONENTS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="红白葡萄酒分类：PCA/LDA + Logistic Regression")
    parser.add_argument("--backend", choices=("all", "numpy", "mindspore"), default="all")
    parser.add_argument("--reducer", choices=("all", "pca", "lda"), default="all")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def selected_backends(name: str) -> List[str]:
    return ["numpy", "mindspore"] if name == "all" else [name]


def selected_reducers(name: str) -> List[str]:
    return ["pca", "lda"] if name == "all" else [name]


def print_results(results: List[ExperimentResult]) -> None:
    print("\n实验结果")
    print("-" * 78)
    print(f"{'Backend':<12} {'Reducer':<8} {'Dims':>4} {'Train Acc':>12} {'Test Acc':>12} {'Final Loss':>12}")
    print("-" * 78)
    for result in results:
        print(
            f"{result.backend:<12} "
            f"{result.reducer:<8} "
            f"{result.n_components:>4d} "
            f"{result.train_accuracy:>12.4f} "
            f"{result.test_accuracy:>12.4f} "
            f"{result.final_loss:>12.6f}"
        )
    print("-" * 78)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs 必须大于 0")
    if args.lr <= 0.0:
        raise ValueError("--lr 必须大于 0")

    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)

    print("=" * 78)
    print("红白葡萄酒分类：PCA/LDA 降维 + 逻辑回归")
    print("=" * 78)
    print(f"数据目录: {data_dir}")
    print(f"随机种子: {args.seed}")
    print(f"测试集比例: {args.test_size}")
    print(f"PCA 降维维度: {PCA_COMPONENTS}")
    print(f"逻辑回归: epochs={args.epochs}, lr={args.lr}")

    split = load_train_test_split(data_dir=data_dir, test_size=args.test_size, seed=args.seed)
    print("\n数据集")
    print(f"训练集: {split.x_train.shape}, {label_counts(split.y_train)}")
    print(f"测试集: {split.x_test.shape}, {label_counts(split.y_test)}")

    if not args.no_plots:
        print("\n生成降维可视化图片...")
        plot_paths = save_reduction_plots(
            split=split,
            output_dir=output_dir,
            pca_components=PCA_COMPONENTS,
        )
        for path in plot_paths:
            print(f"已保存: {path}")

    results: List[ExperimentResult] = []
    for backend in selected_backends(args.backend):
        for reducer in selected_reducers(args.reducer):
            print(f"\n运行路径: {backend.upper()} + {reducer.upper()} + Logistic Regression")
            if backend == "numpy":
                result = run_numpy_experiment(
                    reducer_name=reducer,
                    x_train=split.x_train,
                    y_train=split.y_train,
                    x_test=split.x_test,
                    y_test=split.y_test,
                    epochs=args.epochs,
                    lr=args.lr,
                    pca_components=PCA_COMPONENTS,
                )
            else:
                result = run_mindspore_experiment(
                    reducer_name=reducer,
                    x_train=split.x_train,
                    y_train=split.y_train,
                    x_test=split.x_test,
                    y_test=split.y_test,
                    epochs=args.epochs,
                    lr=args.lr,
                    pca_components=PCA_COMPONENTS,
                    seed=args.seed,
                )
            results.append(result)
            print(
                f"完成: 降维后维度={result.n_components}, "
                f"train_acc={result.train_accuracy:.4f}, "
                f"test_acc={result.test_accuracy:.4f}"
            )

    print_results(results)


if __name__ == "__main__":
    main()
