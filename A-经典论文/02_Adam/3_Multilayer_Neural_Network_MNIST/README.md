# 实验 3：MNIST 多层神经网络的优化器对比

## 一、任务理解

本实验对应 Adam 论文第 6.2 节 **Multi-layer Neural Networks**，是本项目 Adam 系列编号的第三个实验。输入是展平后的 MNIST 灰度图像，输出是 0–9 的类别概率。与前面的凸逻辑回归不同，两隐藏层 ReLU 网络的训练目标是非凸的。

论文明确采用两个各含 1000 个单元的全连接隐藏层、ReLU 激活与 128 的小批量。图 2(a) 比较使用 Dropout 时 Adam、AdaGrad、RMSProp、Nesterov SGD 和 AdaDelta 的训练曲线；图 2(b) 在不使用随机正则化、含 L2 惩罚的目标上比较 Adam 与 SFO。

本实现使用 NumPy 手写前向传播、反向传播和五种一阶优化器，提供两组实验：

- **L2 组**：关闭 Dropout，使用交叉熵加权重 L2 惩罚，比较五种一阶优化器。这是对确定性目标的补充对照，**不是原论文的 Adam–SFO 对比**。
- **Dropout 组**：输入层及两个隐藏层加 Dropout，不额外添加 L2 惩罚，比较图 2(a) 的五种方法。

默认缩小网络和数据量，目标是观察非凸目标下的优化进展，而非匹配原图的数值。主要报告关闭 Dropout 后的完整训练集 NLL，并用验证集 NLL 和准确率检查泛化表现。训练速度与分类效果不是同一个指标。

**复现边界**：未实现 SFO，不验证论文关于 SFO 收敛、内存占用或每步慢 5–10 倍的结论；也不实现论文的密集超参数网格搜索。论文正文没有完整交代本节的初始化、各方法最优超参数、Dropout 概率与位置、L2 系数、数据划分及图中加噪损失的评估细节，下面这些具体选择均属于本实现，不应归因于论文。

## 二、理论实现

### 1. 两隐藏层网络

设小批量输入 $X\in\mathbb R^{B\times784}$，隐藏层宽度为 $h_1,h_2$，参数维度为：

$$
W_1\in\mathbb R^{784\times h_1},\quad
W_2\in\mathbb R^{h_1\times h_2},\quad
W_3\in\mathbb R^{h_2\times10}.
$$

偏置分别为 $b_1\in\mathbb R^{h_1}$、$b_2\in\mathbb R^{h_2}$、$b_3\in\mathbb R^{10}$，在批量维广播。没有 Dropout 时：

$$
Z_1=XW_1+b_1,\quad H_1=\max(Z_1,0),
$$
$$
Z_2=H_1W_2+b_2,\quad H_2=\max(Z_2,0),\quad S=H_2W_3+b_3.
$$

对每一行减去最大 logit 后计算 Softmax，避免指数溢出。平均交叉熵即负对数似然：

$$
\ell_{\mathrm{NLL}}=-\frac1B\sum_{i=1}^B\log P_{i,y_i},
\qquad P=\operatorname{softmax}(S).
$$

代码用 log-sum-exp 直接计算 NLL，不通过裁剪概率改变损失。

### 2. L2 惩罚与 Dropout

L2 组训练目标为：

$$
\mathcal L=\ell_{\mathrm{NLL}}+
\frac\lambda2\sum_{l=1}^3\|W_l\|_F^2.
$$

本实现不惩罚偏置；$\lambda W_l$ 加入梯度后再交给优化器。这是耦合 L2 正则，**不是 AdamW 的解耦权重衰减**。论文写的是对参数做 L2 weight decay，本实现仅作用于权重是一个明确的实现选择。

Dropout 采用倒置缩放。丢弃率为 $p$，独立掩码为 $M_{ij}\sim\operatorname{Bernoulli}(1-p)$，则：

$$
\widetilde H=H\odot D,\qquad D=\frac M{1-p}.
$$

训练时依次对输入 $X$、ReLU 后的 $H_1$ 和 $H_2$ 加噪；反向传播使用同一次前向生成的掩码。评估时关闭 Dropout，不再缩放激活或权重。默认 Dropout 组的 $\lambda=0$。

### 3. 手写反向传播

令 $Y$ 为 One-hot 标签矩阵，输出层误差信号为：

$$
\Delta_3=(P-Y)/B,\qquad
\nabla W_3=\widetilde H_2^\top\Delta_3+\lambda W_3.
$$

继续向前两层传播：

$$
\Delta_2=(\Delta_3W_3^\top)\odot D_2\odot\mathbf1[Z_2>0],
\qquad \nabla W_2=\widetilde H_1^\top\Delta_2+\lambda W_2,
$$
$$
\Delta_1=(\Delta_2W_2^\top)\odot D_1\odot\mathbf1[Z_1>0],
\qquad \nabla W_1=\widetilde X^\top\Delta_1+\lambda W_1.
$$

各层偏置梯度是对应 $\Delta$ 在批量维的求和。不使用 Dropout 时将 $D_1,D_2$ 视为全 1；输入层掩码已经体现在 $\widetilde X$ 中，无需再给权重梯度乘一次。ReLU 在零点的导数取零。

### 4. 五种优化器

以下运算均逐坐标进行，状态从零开始。$g_t$ 包含适用的 L2 梯度，$\alpha$ 是该方法的恒定基础学习率。

**Adam**：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
$$
$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\quad
\hat v_t=\frac{v_t}{1-\beta_2^t},\quad
\theta_t=\theta_{t-1}-\alpha\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

$t$ 从第一次更新开始计数，不在每个 epoch 重置。

**AdaGrad**：

$$
G_t=G_{t-1}+g_t^2,\qquad
\theta_t=\theta_{t-1}-\alpha\frac{g_t}{\sqrt{G_t}+\epsilon}.
$$

**RMSProp**：采用非 centered、无动量版本，稳定项在根号外：

$$
V_t=\rho V_{t-1}+(1-\rho)g_t^2,\qquad
\theta_t=\theta_{t-1}-\alpha\frac{g_t}{\sqrt{V_t}+\epsilon}.
$$

**Nesterov SGD**：速度 $u$ 表示参数位移，梯度在前瞻点求取：

$$
g_t=\nabla\mathcal L_t(\theta_{t-1}+\mu u_{t-1}),\quad
u_t=\mu u_{t-1}-\alpha g_t,\quad
\theta_t=\theta_{t-1}+u_t.
$$

这里 $\mathcal L_t$ 使用当前小批量及其一次性抽取的 Dropout 掩码，L2 梯度也在前瞻点计算。

**AdaDelta**：同时维护梯度平方与未乘外部学习率的更新平方的 EMA：

$$
G_t=\rho G_{t-1}+(1-\rho)g_t^2,\quad
\delta_t=-\frac{\sqrt{U_{t-1}+\epsilon}}{\sqrt{G_t+\epsilon}}g_t,
$$
$$
\theta_t=\theta_{t-1}+\alpha\delta_t,\quad
U_t=\rho U_{t-1}+(1-\rho)\delta_t^2.
$$

默认外部乘子 $\alpha=1$，稳定项在根号内。不要直接把它的学习率设成与 Adam 相同的数值来追求所谓“公平”。

本实现不沿用第 6.1 节逻辑回归的 $1/\sqrt t$ 衰减。恒定学习率及以上优化器变体是本次实现的选择，不声称与作者每项实验的内部实现完全相同。

### 5. 指标与公平性

- 所有组合复制同一份初始参数，优化器状态相互独立。
- 样本排列和 Dropout 使用两个独立随机数生成器；每种方法重新使用相同种子。相同形状、批量和轮数下，Dropout 组的各方法看到相同的掩码序列。
- 每轮末固定参数、关闭 Dropout，按样本加权评估完整训练集和验证集。训练 NLL 不含 L2；`train_objective` 才包含 L2。
- `online_nll`、`online_objective` 是本轮更新前各批次损失的样本加权均值，参数在批次间变化，且可能带 Dropout。Nesterov 的在线值在前瞻点计算，因此不能与轮末完整数据集指标混为一谈。epoch 0 的在线值为空。
- 所有方法完成训练后才评估测试集，不提前停止、不选择最佳 checkpoint、不使用测试集调参。
- `training_seconds` 仅累计小批量循环时间，包含取批次、采样、前后向、数值检查及更新，排除数据加载与整集评估。`elapsed_seconds` 额外包含逐轮评估和训练进度输出，不含测试与绘图。时间曲线不是原论文硬件条件下的耗时复现。

## 三、运行环境

- Python：3.10 或更高版本（代码使用 `X | None` 类型标注）。
- numpy：1.23 或更高版本，用于矩阵运算、随机采样与数组存储。
- matplotlib：3.6 或更高版本，用于曲线绘制和 PNG 输出，并提供代码使用的 `seaborn-v0_8-whitegrid` 样式；无需额外安装 Seaborn。

无需深度学习框架。

## 四、代码说明

### 1. 文件职责

| 文件 | 职责 |
| --- | --- |
| `main.py` | 统一入口，串联配置校验、数据加载、对照训练、测试评估和结果输出。 |
| `config.py` | 集中配置：`EXPERIMENT`（随机种子及各随机流偏移、数据与输出路径、样本量、输入维度与类别数、隐藏层宽度、训练预算、是否显示图像）、`REGIMES`（两组正则化方案）、`OPTIMIZERS`（五种优化器超参数）。 |
| `data.py` | `load_mnist` 管理缓存、抽样与划分；`get_mnist_file` 复用或下载数据；`take_subset` 抽样并缩放像素；`resolve_path` 统一把相对路径解析到本实验目录。 |
| `model.py` | `initialize_parameters` 初始化网络；`apply_dropout`、`forward` 实现倒置 Dropout 与前向传播；`cross_entropy`、`l2_penalty` 计算 NLL 与 L2；`loss_and_gradients` 返回目标值、NLL 和全部参数的解析梯度。 |
| `optimizers.py` | `Optimizer` 根据名称执行五种更新规则；`parameters_for_gradient` 处理 Nesterov 前瞻点；`validate_settings` 检查超参数。 |
| `training.py` | `validate_config` 校验配置；`train_one_optimizer` 执行单次训练并记录逐轮指标；`run_experiments` 组织全部组合及最终测试评估。 |
| `evaluation.py` | `evaluate` 关闭 Dropout，分批计算完整数据集的 NLL、目标值和准确率。 |
| `reporting.py` | `print_final_metrics` 打印最终指标；`save_results` 保存配置与逐轮指标，并绘制训练 NLL、验证 NLL、训练 NLL–时间曲线。 |

### 2. 默认配置与数据路径

默认值如下：

| 项目 | 默认值 |
| --- | --- |
| 随机种子 | 7 |
| 原训练集抽样上限 | 6000，再划出 10% 验证集，即训练 5400、验证 600 |
| 官方测试集抽样上限 | 2000，与训练集独立抽样 |
| 网络 | 784–128–128–10；各层权重采用 He 正态初始化，偏置为零 |
| 训练预算 | 每个组合 10 个 epoch，小批量 128，评估批量 512 |
| L2 组 | $\lambda=10^{-4}$，不使用 Dropout |
| Dropout 组 | 输入丢弃率 0.2、两个隐藏层各 0.5，$\lambda=0$ |
| Adam | $\alpha=10^{-3},\beta_1=0.9,\beta_2=0.999,\epsilon=10^{-8}$ |
| AdaGrad | $\alpha=10^{-2},\epsilon=10^{-8}$ |
| RMSProp | $\alpha=10^{-3},\rho=0.9,\epsilon=10^{-8}$ |
| Nesterov | $\alpha=0.05,\mu=0.9$ |
| AdaDelta | $\alpha=1,\rho=0.95,\epsilon=10^{-6}$ |

像素展平后缩放到 `[0, 1]`，不做白化或数据增强。常驻特征为 `float32`，批次计算、参数和优化器状态为 `float64`，标签为 `int64`。训练/验证使用固定种子的随机划分，不做分层抽样；极小子集不保证覆盖全部类别。

数据来源为 Keras 公共 MNIST NPZ 文件。代码首先查找 `data_dir` 下的 `mnist.npz`，没有时查找 `reuse_data_file` 指向的已有文件（默认指向同系列逻辑回归 MNIST 实验已缓存的同一数据文件，仅读取数据，不导入其代码）；两者都不存在才在用户运行时下载到 `data_dir`。下载先写 `.part` 临时文件，完成后替换，失败时清理临时文件。已有文件不重复下载，损坏缓存会报错，不自动覆盖。

`data_dir`、`reuse_data_file` 和 `output_dir` 的相对路径均以本实验目录为基准，不随启动目录变化。可将 `reuse_data_file` 改为其他已有 MNIST NPZ 的路径，或设为 `None` 禁用复用。各随机流（测试集抽样、验证划分、批次顺序、Dropout 掩码）使用 `seed` 加 `seed_offsets` 中对应偏移量生成。

要扩大到论文网络与图 2(a) 的训练预算，可将 `hidden_sizes` 改为 `(1000, 1000)`，两个 `max_*_samples` 设为 `None`，`epochs` 改为 `200`，保留批量 128。此时仍会从官方 60000 个训练样本中留出 10% 验证集，即实际训练 54000 个；不等同于严格复现。大网络的 NumPy/float64 训练会明显增加时间和内存需求。

所有超参数是未经搜索的教学起点。应在固定验证划分和相同调参预算下分别调整每种方法，再固定配置评估测试集。默认两组共享优化器设置，尚未对每种正则化方案分别调优；没有 SFO、密集网格搜索和完整原始实验细节，即使扩大规模也不能声称完成原图的严格复现。

### 3. main 执行流程

在当前实验目录中执行：

```bash
python main.py
```

实际顺序为：

1. 设置绘图风格，校验配置，再解析路径并读取 MNIST；仅用户主动运行时可能下载数据。
2. 打印实际训练/验证/测试样本数和网络结构。
3. 只生成一份公共初值，依次执行 L2、Dropout 两组中五种方法，共十次训练。
4. 记录 epoch 0 的初始指标；各轮训练后打印训练 NLL、验证 NLL 和验证准确率。发现非有限损失、梯度或参数时明确报错，不跳过失败方法伪造完整曲线。
5. 全部训练结束后评估测试集，打印各组合的最终训练 NLL、验证 NLL/准确率、测试 NLL/准确率和训练循环耗时。
6. 在 `output_dir`（默认 `results`，相对本实验目录）保存 `metrics.json`，包含实际运行配置、软件环境、样本数、指标协议、逐轮历史、测试指标与耗时；不保存模型权重。
7. 在同一目录保存 `training_curves.png`。每组绘制关闭 Dropout 的训练 NLL–epoch、验证 NLL–epoch、训练 NLL–训练时间三张图，NLL 轴采用对数尺度。默认再显示图像；将 `show_plots=False` 可只保存不弹窗。无图形界面的环境也可设置 `MPLBACKEND=Agg`。

重复运行会覆盖同名结果，应自行使用不同 `output_dir` 保留不同配置的输出。导入模块不会主动下载、训练或绘图；入口不包含自动运行的实现测试。

## 五、实验结果

**论文报告**：Dropout 神经网络中 Adam 的训练收敛优于图中其他一阶方法；对于确定性目标，Adam 相对 SFO 在迭代次数和墙钟时间上进展更快。上述为论文结论，不是本次实测结果。

**本次预期**：观察 Adam 是否能在小型非凸网络上较快降低训练 NLL，比较各方法在 Dropout 噪声下的训练进展，并检查训练与验证曲线是否出现过拟合迹象。Dropout 不一定降低训练 NLL；默认规模、单随机种子和未经搜索的学习率也不保证论文中的排序。

**当前状态**：实验代码未运行，结果未实测；未运行梯度检查或优化器数值测试。本次整理没有生成或修改指标、曲线与耗时记录，仅做静态核对与语法检查。不能据此宣称复现成功。
