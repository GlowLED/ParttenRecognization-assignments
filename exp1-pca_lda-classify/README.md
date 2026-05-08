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

## 数学推导

### 1. PCA（主成分分析）

#### 1.1 问题定义

给定 $n$ 个 $d$ 维样本 $\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n \in \mathbb{R}^d$，PCA 的目标是找到一个投影方向 $\mathbf{w} \in \mathbb{R}^d$（$\|\mathbf{w}\| = 1$），使得投影后的数据方差最大。

#### 1.2 目标函数

投影后的样本为 $z_i = \mathbf{w}^T \mathbf{x}_i$，投影后方差为：

$$\text{Var}(z) = \frac{1}{n} \sum_{i=1}^{n} (z_i - \bar{z})^2 = \mathbf{w}^T \mathbf{S} \mathbf{w}$$

其中 $\mathbf{S}$ 为样本协方差矩阵：

$$\mathbf{S} = \frac{1}{n} \sum_{i=1}^{n} (\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^T = \frac{1}{n} \mathbf{X}_c^T \mathbf{X}_c$$

其中 $\mathbf{X}_c$ 为中心化后的数据矩阵。

#### 1.3 优化问题

$$\max_{\mathbf{w}} \quad \mathbf{w}^T \mathbf{S} \mathbf{w}$$
$$\text{s.t.} \quad \mathbf{w}^T \mathbf{w} = 1$$

#### 1.4 求解方法：拉格朗日乘子法

构造拉格朗日函数：

$$L(\mathbf{w}, \lambda) = \mathbf{w}^T \mathbf{S} \mathbf{w} - \lambda (\mathbf{w}^T \mathbf{w} - 1)$$

对 $\mathbf{w}$ 求导并令其为零：

$$\frac{\partial L}{\partial \mathbf{w}} = 2\mathbf{S}\mathbf{w} - 2\lambda\mathbf{w} = 0$$

$$\mathbf{S}\mathbf{w} = \lambda\mathbf{w}$$

这是一个特征值问题。方差为：

$$\mathbf{w}^T \mathbf{S} \mathbf{w} = \mathbf{w}^T \lambda \mathbf{w} = \lambda$$

因此，**最大化方差等价于选择最大特征值对应的特征向量**。

#### 1.5 SVD 求解

实际计算中，对中心化矩阵 $\mathbf{X}_c$ 进行奇异值分解（SVD）：

$$\mathbf{X}_c = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$

协方差矩阵为：

$$\mathbf{S} = \frac{1}{n} \mathbf{V} \mathbf{\Sigma}^2 \mathbf{V}^T$$

特征值 $\lambda_i = \sigma_i^2 / n$，特征向量即为 $\mathbf{V}$ 的列向量。选择前 $k$ 个最大特征值对应的特征向量构成投影矩阵 $\mathbf{W}_k \in \mathbb{R}^{d \times k}$。

#### 1.6 降维

$$\mathbf{Z} = \mathbf{X}_c \mathbf{W}_k$$

其中 $\mathbf{Z} \in \mathbb{R}^{n \times k}$ 为降维后的数据。

#### 1.7 方差解释比

第 $i$ 个主成分的方差解释比为：

$$\text{ratio}_i = \frac{\lambda_i}{\sum_{j=1}^{d} \lambda_j} = \frac{\sigma_i^2}{\sum_{j=1}^{d} \sigma_j^2}$$

---

### 2. LDA（线性判别分析）

#### 2.1 问题定义

LDA（Fisher's Linear Discriminant）的目标是找到投影方向 $\mathbf{w}$，使得投影后**类间方差最大化**，**类内方差最小化**。

#### 2.2 统计量定义

- **类内散布矩阵**（Within-class scatter matrix）：

$$\mathbf{S}_W = \sum_{k=1}^{K} \sum_{\mathbf{x}_i \in C_k} (\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^T$$

- **类间散布矩阵**（Between-class scatter matrix）：

$$\mathbf{S}_B = \sum_{k=1}^{K} n_k (\boldsymbol{\mu}_k - \boldsymbol{\mu})(\boldsymbol{\mu}_k - \boldsymbol{\mu})^T$$

其中：
- $K$ 为类别数
- $C_k$ 为第 $k$ 类样本集合
- $n_k = |C_k|$ 为第 $k$ 类样本数
- $\boldsymbol{\mu}_k$ 为第 $k$ 类样本均值
- $\boldsymbol{\mu}$ 为所有样本的全局均值

#### 2.3 Fisher 准则函数

$$J(\mathbf{w}) = \frac{\mathbf{w}^T \mathbf{S}_B \mathbf{w}}{\mathbf{w}^T \mathbf{S}_W \mathbf{w}}$$

#### 2.4 优化问题

$$\max_{\mathbf{w}} \quad J(\mathbf{w}) = \frac{\mathbf{w}^T \mathbf{S}_B \mathbf{w}}{\mathbf{w}^T \mathbf{S}_W \mathbf{w}}$$

#### 2.5 求解方法：广义特征值问题

对 $J(\mathbf{w})$ 求导并令其为零：

$$\frac{\partial J}{\partial \mathbf{w}} = \frac{2\mathbf{S}_B \mathbf{w} (\mathbf{w}^T \mathbf{S}_W \mathbf{w}) - 2\mathbf{S}_W \mathbf{w} (\mathbf{w}^T \mathbf{S}_B \mathbf{w})}{(\mathbf{w}^T \mathbf{S}_W \mathbf{w})^2} = 0$$

化简得：

$$\mathbf{S}_B \mathbf{w} = \lambda \mathbf{S}_W \mathbf{w}$$

其中 $\lambda = \frac{\mathbf{w}^T \mathbf{S}_B \mathbf{w}}{\mathbf{w}^T \mathbf{S}_W \mathbf{w}}$。

#### 2.6 正则化求解

当 $\mathbf{S}_W$ 奇异时，添加正则化项：

$$\mathbf{S}_W^* = \mathbf{S}_W + \alpha \mathbf{I}$$

其中 $\alpha > 0$ 为正则化系数（本实验取 $\alpha = 10^{-4}$）。

求解广义特征值问题：

$$\mathbf{S}_W^{*-1} \mathbf{S}_B \mathbf{w} = \lambda \mathbf{w}$$

#### 2.7 降维维度

对于 $K$ 类问题，LDA 最多可提取 $K-1$ 个有效判别方向。选择前 $k$ 个最大特征值对应的特征向量构成投影矩阵 $\mathbf{W}_k \in \mathbb{R}^{d \times k}$（$k \leq K-1$）。

#### 2.8 降维

$$\mathbf{Z} = (\mathbf{X} - \boldsymbol{\mu}) \mathbf{W}_k$$

---

### 3. Softmax Regression（多项逻辑回归）

#### 3.1 模型定义

对于 $K$ 类分类问题，给定输入特征 $\mathbf{x} \in \mathbb{R}^d$，模型参数 $\mathbf{W} \in \mathbb{R}^{d \times K}$ 和偏置 $\mathbf{b} \in \mathbb{R}^K$。

线性变换（logits）：

$$\mathbf{z} = \mathbf{W}^T \mathbf{x} + \mathbf{b}$$

#### 3.2 Softmax 函数

将 logits 转换为概率分布：

$$P(y = k | \mathbf{x}) = \hat{p}_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

**数值稳定性优化**：为防止指数溢出，先减去最大值：

$$z_k' = z_k - \max_j z_j$$

$$\hat{p}_k = \frac{e^{z_k'}}{\sum_{j=1}^{K} e^{z_j'}}$$

#### 3.3 交叉熵损失

对于单个样本 $(\mathbf{x}_i, y_i)$，其中 $y_i \in \{0, 1, \ldots, K-1\}$：

$$\ell_i = -\log \hat{p}_{y_i} = -z_{y_i} + \log \sum_{j=1}^{K} e^{z_j}$$

对于 $N$ 个样本的平均损失：

$$L(\mathbf{W}, \mathbf{b}) = -\frac{1}{N} \sum_{i=1}^{N} \log \hat{p}_{y_i}$$

#### 3.4 梯度推导

定义 one-hot 编码标签 $\mathbf{t}_i \in \{0, 1\}^K$，其中 $t_{i,k} = \mathbb{1}[y_i = k]$。

对 logits $z_k$ 求导：

$$\frac{\partial \ell_i}{\partial z_k} = \hat{p}_k - t_{i,k}$$

对权重 $\mathbf{W}$ 求导（利用链式法则）：

$$\frac{\partial \ell_i}{\partial \mathbf{W}} = \mathbf{x}_i (\hat{\mathbf{p}}_i - \mathbf{t}_i)^T$$

对偏置 $\mathbf{b}$ 求导：

$$\frac{\partial \ell_i}{\partial \mathbf{b}} = \hat{\mathbf{p}}_i - \mathbf{t}_i$$

对于整个数据集的平均梯度：

$$\nabla_{\mathbf{W}} L = \frac{1}{N} \mathbf{X}^T (\hat{\mathbf{P}} - \mathbf{T})$$

$$\nabla_{\mathbf{b}} L = \frac{1}{N} \sum_{i=1}^{N} (\hat{\mathbf{p}}_i - \mathbf{t}_i)$$

其中 $\hat{\mathbf{P}} \in \mathbb{R}^{N \times K}$ 为预测概率矩阵，$\mathbf{T} \in \mathbb{R}^{N \times K}$ 为 one-hot 标签矩阵。

---

### 4. 优化方法

#### 4.1 全批量梯度下降（Batch Gradient Descent）

参数更新规则：

$$\mathbf{W}^{(t+1)} = \mathbf{W}^{(t)} - \eta \nabla_{\mathbf{W}} L$$

$$\mathbf{b}^{(t+1)} = \mathbf{b}^{(t)} - \eta \nabla_{\mathbf{b}} L$$

其中 $\eta$ 为学习率（本实验取 $\eta = 0.1$）。

#### 4.2 早停策略（Early Stopping）

为防止过拟合和不必要的计算，采用基于 loss 变化的早停策略：

```
stale_epochs = 0
previous_loss = None

for epoch in 1, 2, ..., max_epochs:
    loss = train_one_epoch()
    
    if previous_loss is not None and |previous_loss - loss| < tol:
        stale_epochs += 1
        if stale_epochs >= patience:
            converged = True
            break
    else:
        stale_epochs = 0
    
    previous_loss = loss
```

参数说明：
- `tol = 1e-7`：loss 变化阈值
- `patience = 20`：连续满足阈值的轮数
- `max_epochs = 100`：最大迭代轮数

#### 4.3 收敛性分析

梯度下降的收敛速度取决于：

1. **学习率 $\eta$**：过大导致震荡，过小导致收敛慢
2. **损失函数的 Lipschitz 常数**：$L$-smoothness 条件
3. **初始化**：零初始化 vs 随机初始化

对于凸函数，梯度下降的收敛速率为 $O(1/t)$，其中 $t$ 为迭代次数。

---

### 5. 算法复杂度分析

| 算法 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| PCA (SVD) | $O(nd \min(n, d))$ | $O(nd + d^2)$ |
| LDA | $O(nd^2 + d^3)$ | $O(d^2)$ |
| Softmax Regression (per epoch) | $O(ndK)$ | $O(dK + nK)$ |

其中 $n$ 为样本数，$d$ 为特征维度，$K$ 为类别数。

---

### 6. 优化技巧

#### 6.1 数值稳定性

- **Softmax**：减去最大值防止指数溢出
- **对数计算**：添加小量 $\epsilon = 10^{-8}$ 防止 $\log(0)$
- **LDA 正则化**：添加 $\alpha \mathbf{I}$ 防止 $\mathbf{S}_W$ 奇异

#### 6.2 数据预处理

- **标准化**：$\mathbf{x}' = (\mathbf{x} - \boldsymbol{\mu}) / \boldsymbol{\sigma}$，使各特征尺度一致
- **中心化**：PCA 要求数据零均值

#### 6.3 初始化策略

- **零初始化**：本实验采用，适合线性模型
- **Xavier 初始化**：适合深度网络，$\mathbf{W} \sim \mathcal{N}(0, 2/(d_{in} + d_{out}))$

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
