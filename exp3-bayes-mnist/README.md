# Exp3: 朴素贝叶斯 MNIST 分类实验

## 1. 实验目的

- 理解朴素贝叶斯分类器的原理，特别是BernoulliNB模型
- 独立实现BernoulliNB算法（使用NumPy）
- 基于MindSpore平台实现相同的算法
- 对比NumPy独立实现、MindSpore平台实现与sklearn官方实现的差异
- 使用Fashion-MNIST数据集进行额外测试（加分项）

## 2. 算法原理

### 2.1 朴素贝叶斯理论

朴素贝叶斯分类器基于贝叶斯定理和特征条件独立假设：

**贝叶斯定理：**
$$P(y=k | \mathbf{x}) = \frac{P(\mathbf{x} | y=k) P(y=k)}{P(\mathbf{x})}$$

**朴素假设（特征条件独立）：**
$$P(\mathbf{x} | y=k) = \prod_{i=1}^{n} P(x_i | y=k)$$

**分类决策：**
$$\hat{y} = \arg\max_{k} P(y=k) \prod_{i=1}^{n} P(x_i | y=k)$$

### 2.2 BernoulliNB模型

BernoulliNB适用于**二值特征**（特征取值为0或1），其似然函数为：

$$P(x_i | y=k) = \begin{cases} P(x_i=1 | y=k) & \text{if } x_i = 1 \\ 1 - P(x_i=1 | y=k) & \text{if } x_i = 0 \end{cases}$$

**参数估计（带Laplace平滑）：**

- 先验概率：$$P(y=k) = \frac{N_k + \alpha}{N + K\alpha}$$

- 条件概率：$$P(x_i=1 | y=k) = \frac{N_{ki} + \alpha}{N_k + 2\alpha}$$

其中：
- $N$ 为总样本数
- $N_k$ 为类别k的样本数
- $N_{ki}$ 为类别k中特征i为1的样本数
- $\alpha$ 为Laplace平滑系数（本实验取1.0）
- $K$ 为类别数

**对数后验概率（避免下溢）：**

$$\log P(y=k | \mathbf{x}) \propto \log P(y=k) + \sum_{i=1}^{n} [x_i \log P(x_i=1|y=k) + (1-x_i) \log P(x_i=0|y=k)]$$

## 3. 实验环境

- **操作系统**: Linux (Ubuntu)
- **Python**: 3.12
- **MindSpore**: >= 2.8
- **NumPy**: >= 1.26
- **scikit-learn**: >= 1.0

## 4. 实验设置

### 4.1 数据集

| 数据集 | 训练集 | 测试集 | 特征维度 | 类别数 |
|--------|--------|--------|----------|--------|
| MNIST | 60,000 | 10,000 | 784 (28×28) | 10 |
| Fashion-MNIST | 60,000 | 10,000 | 784 (28×28) | 10 |

### 4.2 参数设置

- **Laplace平滑系数**: α = 1.0
- **二值化阈值**: 0.5（像素值 > 0.5 设为1，否则为0）
- **数值精度**: 
  - NumPy实现: float64
  - MindSpore实现: float32
  - sklearn实现: float64

## 5. 实验结果

### 5.1 MNIST数据集结果

| Backend | 训练准确率 | 测试准确率 | 训练时间(s) | 预测时间(s) |
|---------|-----------|-----------|-------------|-------------|
| NumPy | 0.8358 | 0.8427 | 0.6888 | 1.5137 |
| MindSpore | 0.8358 | 0.8427 | 4.9936 | 0.4692 |
| sklearn | 0.8358 | 0.8427 | 3.1680 | 0.2533 |

### 5.2 Fashion-MNIST数据集结果

| Backend | 训练准确率 | 测试准确率 | 训练时间(s) | 预测时间(s) |
|---------|-----------|-----------|-------------|-------------|
| NumPy | 0.6511 | 0.6480 | 0.4328 | 1.4758 |
| MindSpore | 0.6511 | 0.6480 | 0.5891 | 0.3772 |
| sklearn | 0.6511 | 0.6480 | 1.8164 | 0.2556 |

### 5.3 结果分析

#### 准确率分析

1. **三种实现的准确率完全一致**：这验证了我们的NumPy和MindSpore实现与sklearn官方实现在数学上是等价的。

2. **MNIST vs Fashion-MNIST**：
   - MNIST测试准确率: **84.27%**
   - Fashion-MNIST测试准确率: **64.80%**
   - 差异: **19.47%**

   Fashion-MNIST准确率显著低于MNIST，原因如下：
   - Fashion-MNIST的类别间相似度更高（如T-shirt/Shirt, Pullover/Coat）
   - 朴素贝叶斯的特征独立假设在复杂图像上更不成立
   - 二值化处理丢失了大量纹理和边缘信息

#### 训练时间分析

| Backend | MNIST训练时间 | Fashion-MNIST训练时间 | 平均训练时间 |
|---------|---------------|----------------------|--------------|
| NumPy | 0.6888s | 0.4328s | **0.5608s** |
| MindSpore | 4.9936s | 0.5891s | 2.7914s |
| sklearn | 3.1680s | 1.8164s | 2.4922s |

**分析：**
- **NumPy实现训练最快**：直接使用NumPy矩阵运算，没有额外开销
- **MindSpore实现较慢**：主要是tensor创建和操作的overhead，特别是在首次运行时
- **sklearn实现居中**：sklearn内部使用了优化的Cython实现

#### 预测时间分析

| Backend | MNIST预测时间 | Fashion-MNIST预测时间 | 平均预测时间 |
|---------|---------------|----------------------|--------------|
| NumPy | 1.5137s | 1.4758s | 1.4948s |
| MindSpore | 0.4692s | 0.3772s | 0.4232s |
| sklearn | 0.2533s | 0.2556s | **0.2545s** |

**分析：**
- **sklearn预测最快**：使用了高度优化的C代码
- **MindSpore预测居中**：tensor运算在大规模矩阵上有优势
- **NumPy预测最慢**：纯Python循环较多

## 6. 差异分析

### 6.1 数值精度差异

- **NumPy/sklearn**: 使用float64（双精度），约15-16位有效数字
- **MindSpore**: 使用float32（单精度），约7位有效数字

在本实验中，由于BernoulliNB的计算相对简单，float32和float64的精度差异未导致准确率差异。但在更复杂的数值计算中，精度差异可能导致结果偏差。

### 6.2 计算效率差异

**训练阶段：**
- NumPy直接使用BLAS/LAPACK优化的矩阵运算
- MindSpore需要创建tensor对象、管理计算图，有额外开销
- sklearn使用Cython实现，针对小规模数据有优化

**预测阶段：**
- MindSpore的tensor运算在大规模矩阵上更高效
- sklearn的预测代码高度优化，使用了向量化操作

### 6.3 实现细节差异

| 实现 | Laplace平滑 | 对数计算 | 内存使用 |
|------|-------------|----------|----------|
| NumPy | 直接实现 | np.log | 低 |
| MindSpore | 直接实现 | np.log（转换回numpy） | 中 |
| sklearn | 内部实现 | 内部优化 | 低 |

## 7. MindSpore使用心得与建议

### 7.1 优势

1. **统一的tensor API**：MindSpore提供了类似PyTorch的tensor操作API，易于上手
2. **自动微分能力**：虽然朴素贝叶斯不需要反向传播，但MindSpore的自动微分对深度学习任务非常有用
3. **跨平台部署**：MindSpore支持多种硬件后端（CPU、GPU、Ascend），便于模型部署
4. **数据管道**：`mindspore.dataset` API提供了高效的数据加载和预处理能力

### 7.2 局限性

1. **传统ML算法支持有限**：MindSpore主要面向深度学习，没有内置的传统机器学习算法（如朴素贝叶斯、SVM等）
2. **小规模数据overhead较大**：对于小规模数据，tensor创建和计算图管理的开销相对较大
3. **文档和示例相对较少**：相比PyTorch和TensorFlow，MindSpore的社区资源和文档相对较少
4. **调试不便**：在PYNATIVE_MODE下调试相对方便，但仍不如纯NumPy灵活

### 7.3 使用建议

1. **深度学习任务优先选择MindSpore**：对于CNN、RNN等深度学习模型，MindSpore是很好的选择
2. **传统ML任务建议使用sklearn**：对于朴素贝叶斯、决策树等传统算法，sklearn更成熟高效
3. **学习tensor操作的工具**：MindSpore可以作为学习tensor操作和深度学习框架的工具
4. **注意数值精度**：MindSpore默认使用float32，在需要高精度的场景下需要注意

### 7.4 与PyTorch/TensorFlow对比

| 特性 | MindSpore | PyTorch | TensorFlow |
|------|-----------|---------|------------|
| API风格 | 类PyTorch | 原生 | 类Keras |
| 静态图支持 | 是 | 是（JIT） | 是 |
| 动态图支持 | 是（PYNATIVE） | 是 | 是（Eager） |
| 硬件支持 | CPU/GPU/Ascend | CPU/GPU | CPU/GPU/TPU |
| 社区活跃度 | 中 | 高 | 高 |
| 传统ML支持 | 弱 | 弱 | 弱 |

## 8. 结论

1. **算法正确性验证**：三种实现（NumPy、MindSpore、sklearn）在相同参数下产生了完全一致的准确率，验证了独立实现的正确性。

2. **朴素贝叶斯的适用场景**：
   - 适合特征独立性假设较强的场景
   - 对于MNIST等相对简单的数据集表现尚可（84.27%）
   - 对于Fashion-MNIST等复杂数据集表现较差（64.80%）
   - 训练和预测速度快，适合大规模数据

3. **MindSpore平台评价**：
   - 适合深度学习任务，提供了完整的训练和部署工具链
   - 对传统机器学习算法支持有限，需要自行实现
   - 在小规模数据上overhead较大，不推荐用于传统ML任务
   - 可作为学习tensor操作和深度学习的平台

4. **性能对比**：
   - 训练速度：NumPy > sklearn > MindSpore
   - 预测速度：sklearn > MindSpore > NumPy
   - 内存使用：NumPy ≈ sklearn < MindSpore

## 9. 文件说明

```
exp3-bayes-mnist/
├── data.py              # 数据下载与加载
├── model.py             # 三种BernoulliNB实现
├── main.py              # 实验主程序
├── tools/
│   └── fetch_official_models.py  # MindSpore官方模型获取脚本
├── data/                # 数据目录（已gitignore）
│   ├── mnist/           # MNIST数据
│   └── fashion_mnist/   # Fashion-MNIST数据
├── outputs/             # 输出目录（已gitignore）
│   ├── summary_metrics.csv           # 指标汇总
│   ├── experiment_results.json       # 完整JSON结果
│   ├── mnist_numpy_predictions.csv   # MNIST NumPy预测
│   ├── mnist_mindspore_predictions.csv
│   ├── mnist_sklearn_predictions.csv
│   ├── fashion_mnist_numpy_predictions.csv
│   ├── fashion_mnist_mindspore_predictions.csv
│   └── fashion_mnist_sklearn_predictions.csv
└── README.md            # 本实验报告
```

## 10. 参考资料

1. MindSpore官方文档: https://www.mindspore.cn/
2. MindSpore模型仓库: https://gitee.com/mindspore/models
3. scikit-learn BernoulliNB文档: https://scikit-learn.org/stable/modules/naive_bayes.html#bernoulli-naive-bayes
4. MNIST数据集: http://yann.lecun.com/exdb/mnist/
5. Fashion-MNIST数据集: https://github.com/zalandoresearch/fashion-mnist
