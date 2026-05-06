"""
CIFAR-10 神经网络训练脚本
纯 NumPy 实现，从头训练多层感知机
"""

import os
import numpy as np
from tqdm import tqdm
from data import CIFAR10Loader, DataLoader, one_hot_encode
from model import MLP

# 训练超参数
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.01

# 模型保存路径
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'mlp_cifar10.pkl')


def train():
    print("=" * 50)
    print("CIFAR-10 神经网络训练")
    print("=" * 50)

    # 加载数据
    print("\n[1/4] 加载 CIFAR-10 数据集...")
    loader = CIFAR10Loader()
    x_train, y_train, x_test, y_test = loader.load_data()
    print(f"  训练集: {x_train.shape[0]} 样本")
    print(f"  测试集: {x_test.shape[0]} 样本")

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

    for epoch in range(EPOCHS):
        # 创建数据加载器 (shuffle=True)
        train_loader = DataLoader(x_train, y_train_onehot, batch_size=BATCH_SIZE, shuffle=True)

        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", unit="batch", ncols=80)
        for batch_x, batch_y in pbar:
            # 前向传播
            pred = model.forward(batch_x)

            # 计算损失
            loss = model.loss_fn(pred, batch_y)
            epoch_loss += loss
            num_batches += 1

            # 反向传播
            grad = model.loss_fn.backpropagation(lr=LEARNING_RATE)
            model.backward(grad, LEARNING_RATE)

            # 更新进度条描述
            pbar.set_postfix({"loss": f"{loss:.4f}"})

        avg_loss = epoch_loss / num_batches

        # 在测试集上评估准确率
        test_pred = model.forward(x_test)
        test_acc = np.mean(np.argmax(test_pred, axis=1) == y_test)

        print(f"Epoch {epoch + 1:2d}/{EPOCHS} | Loss: {avg_loss:.4f} | Test Acc: {test_acc:.4f}")

    print("-" * 50)
    print("训练完成!")

    # 保存模型
    print("\n[4/4] 保存模型...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"模型已保存至: {MODEL_PATH}")


if __name__ == '__main__':
    train()
