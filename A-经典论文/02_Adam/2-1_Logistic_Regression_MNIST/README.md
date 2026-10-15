# 实验 2-1：MNIST 稠密输入的逻辑回归优化器对比

## 一、任务理解

复现 Adam 论文第 6.1 节的 MNIST 凸逻辑回归实验。用于检验 Adam 在常规问题上是否有效。

把 $28\times28$ 图像展平为 784 维，训练带 $L^2$ 正则的 10 类 Softmax 回归，比较 Adam、AdaGrad 与 Nesterov SGD。

三种方法使用相同的初始化、批次顺序和训练预算；小批量为 128，基础学习率按 $\alpha_t=\alpha/\sqrt t$ 衰减。本实验不使用输入 Dropout。目标是通过实现测试并生成训练 NLL 曲线，检查论文报告的方向性现象：Adam 与调好的 Nesterov SGD 接近且快于 AdaGrad。

## 二、理论实现

### 对于单个样本

把一张 $28\times28$ 的图片展开成长度为 $d=784$ 的**列向量** $x\in\mathbb{R}^{d}$，真实类别为 $y\in\{0,\ldots,C-1\}$，其中 $C=10$。

设权重 $W\in\mathbb{R}^{d\times C}$、偏置 $b\in\mathbb{R}^{C}$，则各类别的得分为

$$
z=W^\top x+b\in\mathbb{R}^{C}
$$

通过 Softmax 得到预测概率：

$$
p_c
=
\frac{\exp(z_c-m)}
{\sum_{k=0}^{C-1}\exp(z_k-m)},
\qquad
m=\max_{0\le j<C}z_j
$$

其中 $p_c$ 表示该图片属于类别 $c$ 的预测概率。减去最大值 $m$ 是为了提高数值稳定性，不改变概率结果。

带 $L^2$ 正则的单样本损失为

$$
\mathcal{L}
=
-\log p_y
+
\frac{\lambda}{2}\|W\|_F^2
$$

其中 $p_y$ 是真实类别对应的预测概率，且

$$
\|W\|_F^2
=
\sum_{j=1}^{d}\sum_{c=0}^{C-1}W_{jc}^{2}
$$

若 $q\in\mathbb{R}^{C}$ 是真实标签的 One-hot 向量，即

$$
q_c=
\begin{cases}
1,&c=y,\\
0,&c\ne y,
\end{cases}
$$

则损失也可以写成

$$
\mathcal{L}
=
-\sum_{c=0}^{C-1}q_c\log p_c
+
\frac{\lambda}{2}\|W\|_F^2
$$

相应的梯度为

$$
\frac{\partial\mathcal{L}}{\partial W}
=
x(p-q)^\top+\lambda W,
\qquad
\frac{\partial\mathcal{L}}{\partial b}
=
p-q
$$

其中 $x(p-q)^\top\in\mathbb{R}^{d\times C}$，与权重矩阵 $W$ 的形状一致。正则项只作用于权重，不作用于偏置；若单独记录不含正则项的数据损失，则为

$$
\ell_{\mathrm{NLL}}=-\log p_y
$$

### 对于样本的批量处理

给定小批量 $X\in\mathbb R^{B\times d}$、权重 $W\in\mathbb R^{d\times C}$、偏置 $b\in\mathbb R^C$，Softmax 回归为

$$
Z=XW+b,\qquad
p_{ic}=\frac{\exp(Z_{ic}-\max_j Z_{ij})}{\sum_k\exp(Z_{ik}-\max_j Z_{ij})}.
$$

带 $L^2$ 正则的平均损失为

$$
\mathcal L
=-\frac1B\sum_{i=1}^{B}\log p_{i,y_i}
+\frac\lambda2\lVert W\rVert_2^2.
$$

若 $Y$ 是标签的 One-hot 矩阵，则

$$
\frac{\partial\mathcal L}{\partial W}
=X^\top\frac{(P-Y)}B+\lambda W,
\qquad
\frac{\partial\mathcal L}{\partial b}
=\sum_i\frac{(P-Y)_i}{B}.
$$

本实验中 $X$ 是缩放到 $[0,1]$ 的稠密像素矩阵，$d=784$、$C=10$。正则项只作用于权重，不作用于偏置；训练曲线单独记录不含正则项的数据 NLL。

## 三、运行环境

- Python：3.10 或更高版本。
- numpy：1.23 或更高版本，用于数据处理、矩阵运算与数值测试。
- matplotlib：3.6 或更高版本，用于绘图，并提供代码使用的 `seaborn-v0_8-whitegrid` 样式；无需额外安装 Seaborn。

无需深度学习框架。

## 四、代码说明

### 1. 文件职责

本实验以 `main.py` 为统一入口，按数据处理、模型计算、优化器更新、训练与评估拆分模块。模型和优化器均使用 NumPy 手工实现，不依赖自动微分。

| 文件              | 主要职责                                                                                                            |
| --------------- | --------------------------------------------------------------------------------------------------------------- |
| `main.py`       | 组织完整实验。打印最终指标并绘制图像。                                                                                             |
| `config.py`     | 实验配置。`EXPERIMENT` 管理随机种子、数据路径、批量大小、验证比例、样本量、epoch 数和 L2 系数；`MNIST_OPTIMIZERS` 管理各优化器的基础学习率及专属超参数。               |
| `data.py`       | 下载和加载 MNIST，将图像展平为 784 维并缩放到 `[0, 1]`；提供固定种子抽样、训练/验证划分及批次构造。常驻特征使用 `float32`，计算批次转为 `float64`，标签使用 `int64`。     |
| `model.py`      | 实现参数初始化、数值稳定的 Softmax、带 L2 正则的损失及解析梯度，以及类别概率预测。`softmax_loss_and_gradients` 同时返回总损失、未正则化 NLL 和梯度字典。             |
| `optimizers.py` | 定义统一的 `Optimizer` 接口，手写 `Adam`、`AdaGrad` 和 `NesterovSGD`，并通过 `create_optimizer` 按名称创建实例。优化器负责参数更新，学习率调度由训练循环负责。 |
| `training.py`   | `train_one_optimizer` 执行单个方法的小批量训练，记录逐 epoch 指标和耗时；`run_optimizer_suite` 为三种方法提供相同初始化、数据顺序和训练预算。                |
| `evaluation.py` | `evaluate_logistic_regression` 分批计算完整数据集的 NLL、含 L2 的目标值和准确率。先按样本累加再求平均，避免最后一个较小批次获得错误权重。                        |
| `reporting.py`  | `print_final_metrics` 汇总最终训练 NLL、验证准确率、测试 NLL、测试准确率与耗时；`plot_training_costs` 将三种方法的训练 NLL 绘制在同一坐标轴上。            |

### 2. 默认配置与数据路径

默认配置位于 `config.py`：

- 随机种子为 `7`，批量大小为 `128`，训练 `10` 个 epoch，L2 系数为 `1e-4`。
- 从原训练集抽取 `12,000` 个样本，再划分其中 `10%` 为验证集，因此实际训练集为 `10,800` 个样本、验证集为 `1,200` 个样本；从原测试集独立抽取 `2,000` 个样本。
- Adam 的基础学习率为 `1e-2`，`beta1=0.9`、`beta2=0.999`、`epsilon=1e-8`。
- AdaGrad 的基础学习率为 `1e-1`，`epsilon=1e-8`；Nesterov SGD 的基础学习率为 `1e-1`，`momentum=0.9`。

`data_dir` 默认为 `data/adam`。相对路径以 `main.py` 所在目录为基准，因此默认数据文件为本实验目录下的 `data/adam/mnist.npz`，不随启动时的工作目录变化。首次加载且文件不存在时，代码从 Keras 公共数据地址下载；已有文件会直接复用。下载先写入 `.part` 临时文件，完成后再替换为正式文件。

若要扩大实验规模，可将两个 `max_*_samples` 都设为 `None`，并将 `epochs` 改为 `45`。这只扩大数据量和训练预算，不等于完成论文全部设置的严格复现。当前代码不自动搜索学习率，应分别依据验证集选择各方法的配置，不能根据测试集调参。

### 3. main 执行流程

`main.py` 按以下顺序执行：

1. 设置 NumPy 输出精度与绘图样式，打印 Python、NumPy、Matplotlib 和操作系统版本。
2. 运行微型示例，再运行四项实现测试；每项测试通过后打印对应的 `PASS`。
3. 读取配置，加载并预处理 MNIST，仅从原训练集划分验证集，打印训练/验证/测试样本数。
4. 创建一份公共初始参数，依次训练 Adam、AdaGrad 和 Nesterov SGD。每种方法复制独立的参数副本，并使用同一个 `seed + 100` 生成批次排列，避免初始化或样本顺序影响比较。
5. 每个小批量先计算梯度，再更新参数。Adam 和 AdaGrad 在当前参数处求梯度；Nesterov SGD 在前瞻位置求梯度。三种方法统一使用 $\alpha_t=\alpha/\sqrt t$，其中 $t$ 从第一次参数更新开始计数，跨 epoch 连续累加。
6. 每个 epoch 结束后，在完整训练集和验证集上评估，记录训练 NLL、训练目标值、训练准确率、验证 NLL、验证准确率及该 epoch 最后一步的学习率。
7. 全部训练完成后才评估测试集，打印最终指标，并显示训练 NLL 曲线。

整个流程不使用输入 Dropout，也不根据验证结果提前停止训练。三种方法具有相同的 epoch 数和批量大小，差异来自更新规则及各自配置的超参数。

## 五、实验结果

在常规凸优化任务上，Adam 和 Nesterov SGD 都收敛表现相近，都比 AdaGrad 快。说明 Adam 在常规凸优化任务上表现良好。