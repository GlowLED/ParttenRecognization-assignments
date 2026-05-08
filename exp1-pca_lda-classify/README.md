# Exp1: PCA/LDA 降维与 Logistic Regression 分类实验

## 实验目标

本实验使用 UCI Wine Quality 红白葡萄酒数据集完成二分类任务：红葡萄酒记为 `red`，白葡萄酒记为 `white`。实验分别比较未降维、PCA 降维和 LDA 降维后的 Logistic Regression 分类效果，并对比自己独立实现的 NumPy 算法与基于 MindSpore 官方 models 训练脚本风格改造的实现。

加分项使用 UCI Iris 数据集做三分类任务，复用同一套 NumPy 与 MindSpore 官方风格实验流程。

## 数据与预处理

- Wine 数据集：自动下载 `winequality-red.csv` 和 `winequality-white.csv`，使用 11 个理化指标作为特征，丢弃原始 `quality` 分数列。
- Iris 数据集：自动下载 `iris.data`，使用 4 个花萼/花瓣特征，标签为 3 个鸢尾花类别。
- 所有数据集都使用固定随机种子 `42` 做分层划分，默认测试集比例为 `0.2`。
- 标准化只在训练集上拟合均值和标准差，再应用到训练集和测试集，避免数据泄漏。

## 算法实现

### NumPy 独立实现

- `raw`：不降维，直接在标准化后的原始特征上训练。
- `pca`：手写 PCA，基于训练集 SVD，固定降到 2 维。
- `lda`：手写 Fisher LDA，Wine 二分类降到 1 维，Iris 三分类降到 2 维。
- 分类器：手写多分类 softmax regression。二分类 Wine 也统一视作 2 类 softmax regression。
- 优化：全批量梯度下降，记录 loss、准确率、运行轮数和是否满足早停阈值。

### MindSpore 官方模型库风格实现

官方模型脚本仓库为 `https://gitee.com/mindspore/models`。本实验使用 `tools/fetch_official_models.py` 浅克隆官方仓库的 LeNet 训练/评估脚本作为参考，参考 commit 为 `154d170dbd5740f5c19cb0e3dfe9e88a565d8ba5`。

官方 LeNet 训练脚本采用 `create_dataset -> nn.Cell 网络 -> loss -> optimizer -> Model -> Accuracy/Callback -> model.train()` 的结构。本实验没有直接迁移图像网络结构，而是在同一 Wine/Iris 表格数据上适配该训练范式：

- 数据集：`mindspore.dataset.NumpySlicesDataset`
- 网络：单层 `nn.Dense(input_dim, num_classes)`，等价于 Logistic Regression / Softmax Regression
- 损失：`nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")`
- 优化器：`nn.SGD`
- 训练入口：`mindspore.train.Model`
- 指标和回调：`Accuracy`、`LossMonitor`、自定义收敛回调

这样可以保证 NumPy 与 MindSpore 路径使用相同数据、相同降维结果、相同初始化和相同学习率，便于观察框架实现差异。

## 运行方式

```bash
python exp1-pca_lda-classify/tools/fetch_official_models.py
python exp1-pca_lda-classify/main.py
```

常用参数：

```bash
python exp1-pca_lda-classify/main.py --dataset wine --backend numpy --max-epochs 100
python exp1-pca_lda-classify/main.py --dataset iris --backend mindspore_official --reducer lda
python exp1-pca_lda-classify/main.py --max-epochs 200 --lr 0.1 --tol 1e-7 --patience 20
```

## 输出文件

脚本默认输出到 `exp1-pca_lda-classify/outputs/`：

- `summary_metrics.csv`：所有实验路径的准确率、loss、迭代轮数、耗时。
- `test_predictions.csv`：所有数据集的测试集逐样本分类结果。
- `wine_test_predictions.csv`：Wine 测试集逐样本分类结果。
- `iris_test_predictions.csv`：Iris 测试集逐样本分类结果。
- `experiment_results.json`：完整机器可读结果。
- `wine_pca_2d.png`、`wine_lda_1d.png`：Wine 降维可视化。
- `iris_pca_2d.png`、`iris_lda_2d.png`：Iris 降维可视化。

测试集分类结果文件包含：数据集、backend、降维方式、样本 id、真实标签、预测标签、是否正确、各类别预测概率。

## 实验结果

以下结果来自默认参数：`max_epochs=100`、`lr=0.1`、`tol=1e-7`、`patience=20`。

| 数据集 | Backend | 特征空间 | 维度 | 训练准确率 | 测试准确率 | final loss | 迭代轮数 | 是否收敛 | 耗时(s) |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| wine | NumPy | RAW | 11 | 0.9890 | 0.9892 | 0.087127 | 100 | False | 0.104 |
| wine | NumPy | PCA | 2 | 0.9829 | 0.9838 | 0.100674 | 100 | False | 0.163 |
| wine | NumPy | LDA | 1 | 0.9942 | 0.9954 | 0.130321 | 100 | False | 0.385 |
| wine | MindSporeOfficial | RAW | 11 | 0.9890 | 0.9892 | 0.087127 | 100 | False | 41.669 |
| wine | MindSporeOfficial | PCA | 2 | 0.9829 | 0.9838 | 0.100674 | 100 | False | 40.321 |
| wine | MindSporeOfficial | LDA | 1 | 0.9942 | 0.9954 | 0.130321 | 100 | False | 46.302 |
| iris | NumPy | RAW | 4 | 0.8917 | 0.9333 | 0.323701 | 100 | False | 0.012 |
| iris | NumPy | PCA | 2 | 0.8583 | 0.9000 | 0.345945 | 100 | False | 0.013 |
| iris | NumPy | LDA | 2 | 0.9333 | 0.9333 | 0.369675 | 100 | False | 0.014 |
| iris | MindSporeOfficial | RAW | 4 | 0.8917 | 0.9333 | 0.323701 | 100 | False | 1.862 |
| iris | MindSporeOfficial | PCA | 2 | 0.8583 | 0.9000 | 0.345945 | 100 | False | 1.990 |
| iris | MindSporeOfficial | LDA | 2 | 0.9333 | 0.9333 | 0.369675 | 100 | False | 1.876 |

## 结果分析

Wine 数据集上，未降维 RAW 的测试准确率为 `0.9892`，PCA 降到 2 维后下降到 `0.9838`，说明 PCA 保留了主要方差信息，但方差最大的方向不一定最有利于红白分类。LDA 降到 1 维后测试准确率达到 `0.9954`，高于 RAW 和 PCA，原因是 LDA 使用类别标签优化类间分离度，正好适合红白葡萄酒这种二分类任务。

Iris 数据集上，RAW 和 LDA 的测试准确率均为 `0.9333`，PCA 为 `0.9000`。Iris 的类别结构本身较适合线性分类，LDA 在 2 维投影中保留了类别判别信息，因此降维后没有明显损失；PCA 仍然只关注整体方差，分类效果略低。

NumPy 与 MindSporeOfficial 的准确率和 loss 基本一致，因为两者使用相同的输入特征、相同的全批量梯度下降、相同初始化和相同损失函数。差异主要体现在耗时：小型表格数据上，NumPy 的矩阵运算开销很低，而 MindSpore 的 `Model` 训练流程存在图构建、数据集管线、回调调度等固定开销；这些开销在大型神经网络、批量数据和硬件加速场景下更容易被摊薄。

本次 `converged=False` 表示在 100 轮内 loss 改变量尚未连续满足 `tol=1e-7` 和 `patience=20` 的早停条件，并不代表模型不可用。若需要更严格收敛，可提高 `max_epochs` 或放宽 `tol`。

## MindSpore 使用心得与建议

MindSpore 的 `Model` API 适合组织标准训练流程，数据集、网络、损失、优化器和指标边界清晰，也便于迁移到更复杂模型。官方 models 仓库的脚本结构对工程化实验有参考价值，尤其是把数据构造、网络定义、训练入口和评估入口拆开。

对于本实验这种小型表格任务，直接使用 NumPy 实现更轻量，调试和运行速度更快。使用 MindSpore 时建议：

- 小数据实验优先使用 `PYNATIVE_MODE` 便于调试。
- 如果重点是算法对比，应控制初始化、学习率、batch 策略和降维结果一致。
- 对很小的数据集，不要只用耗时评价框架优劣；MindSpore 的优势更适合大模型、复杂网络和硬件加速场景。
- 官方 models 仓库主要覆盖视觉、NLP、推荐等深度学习模型，表格 Logistic Regression 需要按官方训练范式自行适配。
