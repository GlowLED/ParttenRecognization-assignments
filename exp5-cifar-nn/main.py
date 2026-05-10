"""
CIFAR-10 神经网络训练脚本
支持 NumPy 和 MindSpore 两种后端
"""

import argparse
import json
import os
import time

import numpy as np
from tqdm import tqdm

from data import CIFAR10Loader, DataLoader, one_hot_encode
from model import MLP, MindSporeMLP

# 训练超参数
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.01

# 输出路径
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIFAR-10 神经网络训练")
    parser.add_argument("--backend", choices=["numpy", "mindspore", "all"], default="numpy",
                        help="训练后端: numpy, mindspore, all")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="批大小")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="学习率")
    return parser.parse_args()


def load_data():
    """加载 CIFAR-10 数据集"""
    print("\n[1/4] 加载 CIFAR-10 数据集...")
    loader = CIFAR10Loader()
    x_train, y_train, x_test, y_test = loader.load_data()
    print(f"  训练集: {x_train.shape[0]} 样本")
    print(f"  测试集: {x_test.shape[0]} 样本")
    return x_train, y_train, x_test, y_test


def train_numpy(x_train, y_train, x_test, y_test, epochs, batch_size, lr) -> dict:
    """NumPy 版本训练"""
    print("\n" + "=" * 50)
    print("NumPy 版本训练")
    print("=" * 50)

    # One-hot 编码
    y_train_onehot = one_hot_encode(y_train)
    y_test_onehot = one_hot_encode(y_test)

    # 创建模型
    print("\n[2/4] 创建 MLP 模型...")
    input_dim = 3072  # 32x32x3
    hidden_dims = [512, 256, 128]
    output_dim = 10
    model = MLP(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim)
    print(f"  架构: {input_dim} -> {' -> '.join(map(str, hidden_dims))} -> {output_dim}")

    # 训练循环
    print("\n[3/4] 开始训练...")
    print("-" * 50)

    losses = []
    accuracies = []
    start_time = time.perf_counter()

    for epoch in range(epochs):
        train_loader = DataLoader(x_train, y_train_onehot, batch_size=batch_size, shuffle=True)

        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch", ncols=80)
        for batch_x, batch_y in pbar:
            pred = model.forward(batch_x)
            loss = model.loss_fn(pred, batch_y)
            epoch_loss += loss
            num_batches += 1

            grad = model.loss_fn.backpropagation(lr=lr)
            model.backward(grad, lr)

            pbar.set_postfix({"loss": f"{loss:.4f}"})

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)

        # 测试准确率
        test_pred = model.forward(x_test)
        test_acc = float(np.mean(np.argmax(test_pred, axis=1) == y_test))
        accuracies.append(test_acc)

        print(f"Epoch {epoch + 1:2d}/{epochs} | Loss: {avg_loss:.4f} | Test Acc: {test_acc:.4f}")

    train_time = time.perf_counter() - start_time
    print("-" * 50)
    print("训练完成!")

    # 保存模型
    model_path = os.path.join(MODEL_DIR, 'mlp_cifar10_numpy.pkl')
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(model_path)

    return {
        "backend": "numpy",
        "losses": losses,
        "accuracies": accuracies,
        "epochs_run": epochs,
        "final_accuracy": accuracies[-1],
        "train_time_sec": train_time,
    }


def train_mindspore(x_train, y_train, x_test, y_test, epochs, batch_size, lr) -> dict:
    """MindSpore 版本训练"""
    print("\n" + "=" * 50)
    print("MindSpore 版本训练")
    print("=" * 50)

    input_dim = 3072
    hidden_dims = [512, 256, 128]
    output_dim = 10

    model = MindSporeMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
    )
    print(f"  架构: {input_dim} -> {' -> '.join(map(str, hidden_dims))} -> {output_dim}")

    start_time = time.perf_counter()
    result = model.fit(x_train, y_train, x_test, y_test)
    train_time = time.perf_counter() - start_time

    result["backend"] = "mindspore"
    result["train_time_sec"] = train_time
    return result


def save_results(results: list, output_dir: str):
    """保存实验结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存 JSON 结果
    json_path = os.path.join(output_dir, "experiment_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存至: {json_path}")

    # 打印汇总
    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)
    print(f"{'Backend':<12} {'Final Acc':>10} {'Train Time':>12} {'Epochs':>8}")
    print("-" * 45)
    for r in results:
        print(f"{r['backend']:<12} {r['final_accuracy']:>10.4f} {r['train_time_sec']:>12.2f}s {r['epochs_run']:>8d}")
    print("-" * 45)


def main():
    args = parse_args()

    print("=" * 50)
    print("CIFAR-10 神经网络训练")
    print("=" * 50)
    print(f"Backend: {args.backend}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")

    # 加载数据
    x_train, y_train, x_test, y_test = load_data()

    results = []

    if args.backend in ["numpy", "all"]:
        result = train_numpy(x_train, y_train, x_test, y_test,
                             args.epochs, args.batch_size, args.lr)
        results.append(result)

    if args.backend in ["mindspore", "all"]:
        result = train_mindspore(x_train, y_train, x_test, y_test,
                                 args.epochs, args.batch_size, args.lr)
        results.append(result)

    # 保存结果
    save_results(results, OUTPUT_DIR)


if __name__ == '__main__':
    main()
