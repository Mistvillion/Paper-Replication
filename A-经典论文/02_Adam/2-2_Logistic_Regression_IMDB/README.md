# 实验 2-2：IMDB 稀疏词袋逻辑回归的优化器对比

## 一、任务理解

复现论文第 6.1 节的 IMDB 影评分类实验。输入是一条影评的 10000 维稀疏词袋，输出是影评为正面评价的概率。负面标签为 0，正面标签为 1。固定模型、数据、初始化和训练预算，对比 Adam、AdaGrad、RMSProp 与 Nesterov SGD 四个优化器。

选取最高频的 10000 个词作为词袋特征，训练时施加 50% 输入 Dropout，逻辑回归实验使用随更新步衰减的学习率，主要观察训练负对数似然（NLL）随完整数据遍历次数的变化。

## 二、理论实现

### 1. 稀疏词袋与输入 Dropout

令小批量词袋为 $X\in\mathbb R^{B\times d}$，$d=10,000$。默认使用二值词袋：词出现即记为 1，不出现记为 0，不保留词序，也不使用 TF-IDF 或长度归一化。

输入丢弃概率为 $r=0.5$，本实现采用 inverted Dropout：

$$
M_{ij}\sim\operatorname{Bernoulli}(1-r),\qquad
\widetilde X_{ij}=X_{ij}\frac{M_{ij}}{1-r}.
$$

因此 $\mathbb E[\widetilde X]=X$，评估时直接使用原始 $X$，不再缩放。稀疏矩阵只需对非零元素抽样，零元素始终为零。每次更新都重新生成掩码，各方法共享相同的数据排列和掩码随机序列；掩码施加在词袋坐标上，而非逐个原始词 token 上。

词袋始终使用 SciPy 的 CSR 稀疏存储。权重、梯度和优化器状态是长度为 $d$ 的稠密向量；这不会创建 $N\times d$ 的稠密数据集。

### 2. 二分类逻辑回归与解析梯度

权重 $w\in\mathbb R^d$，偏置 $b\in\mathbb R$，标签 $y_i\in\{0,1\}$：

$$
z=\widetilde Xw+b,\qquad p_i=\sigma(z_i)=\frac1{1+e^{-z_i}}.
$$

带可选 L2 项的训练目标为

$$
\mathcal L=\frac1B\sum_i\log\left(1+\exp((1-2y_i)z_i)\right)
+\frac\lambda2\|w\|_2^2.
$$

实现用 `logaddexp` 计算 NLL，用稳定的 `expit` 计算 sigmoid，避免指数溢出。梯度为

$$
\nabla_w\mathcal L=\widetilde X^\top(p-y)/B+\lambda w,
\qquad
\nabla_b\mathcal L=\frac1B\sum_i(p_i-y_i).
$$

默认 $\lambda=0$，仅使用输入 Dropout 正则；偏置从不参与 L2 正则。二分类逻辑回归使用一个 logit，与两类 Softmax 的概率表达等价，但其参数化与优化轨迹不必相同。零初始化适用于这个线性模型，不存在隐藏层神经元对称性问题。

### 3. 优化器与步长

所有运算逐坐标执行。设 $g_t$ 为当前小批量梯度，$t$ 为从 1 开始、跨 epoch 累加的更新步数，各方法使用各自基础学习率：

$$
\alpha_t=\alpha/\sqrt t.
$$

Adam 显式实现论文 Algorithm 1：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
$$
$$
\hat m_t=m_t/(1-\beta_1^t),\quad
\hat v_t=v_t/(1-\beta_2^t),\quad
\theta_t=\theta_{t-1}-\alpha_t\hat m_t/(\sqrt{\hat v_t}+\epsilon).
$$

AdaGrad 累计平方梯度：

$$
G_t=G_{t-1}+g_t^2,\qquad
\theta_t=\theta_{t-1}-\alpha_t g_t/(\sqrt{G_t}+\epsilon).
$$

RMSProp 使用非中心化的平方梯度 EMA，不加动量、不做偏差修正：

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,\qquad
\theta_t=\theta_{t-1}-\alpha_t g_t/(\sqrt{v_t}+\epsilon).
$$

Nesterov SGD 使用经典前瞻形式，$u_t$ 表示参数位移：

$$
g_t=\nabla\mathcal L_t(\theta_{t-1}+\mu u_{t-1}),\qquad
u_t=\mu u_{t-1}-\alpha_t g_t,\qquad
\theta_t=\theta_{t-1}+u_t.
$$

各方法的状态从零开始。即使本步某坐标梯度为零，Adam 的历史矩、RMSProp 的二阶矩及 Nesterov 的速度也继续按公式更新；没有用只更新非零梯度坐标的稀疏近似替代标准算法。L2 若启用，会直接加入梯度，不是 AdamW 式解耦权重衰减。

### 4. 训练目标与报告指标

- 每轮主曲线：在该轮结束时关闭 Dropout，对完整训练集计算平均 NLL，不含 L2 项；验证 NLL 与准确率也关闭 Dropout。
- 训练目标值：上述确定性 NLL 加 $\lambda\|w\|^2/2$。它不同于对 Dropout 掩码求期望的随机训练目标。
- `dropout_batch_nll`、`dropout_batch_objective`：记录训练期间带噪声的小批量损失，按样本数加权。参数在批间变化，Nesterov 还使用前瞻参数，因此它们只作诊断，不作为主对比曲线。
- 准确率：以 $z\ge0$ 判为正类。NLL 与准确率观察的是不同方面，不能相互替代。

## 三、运行环境

- Python：3.10–3.12
- `Numpy`：1.26 或更新的兼容版本，用于参数、梯度、随机数与数值计算。
- `SciPy`：1.11 或更新的兼容版本，CSR 稀疏矩阵及稳定 sigmoid。
- `Matplotlib`：3.8 或更新的兼容版本，绘制并保存训练与验证曲线。

无需深度学习框架。

## 四、代码说明

### 1. 文件职责

| 文件 | 职责与主要接口 |
| --- | --- |
| `main.py` | `main` 串联配置检查、数据准备、训练、测试评估与结果输出；有入口保护。 |
| `config.py` | `EXPERIMENT` 集中定义实验参数，`IMDB_OPTIMIZERS` 定义各优化器超参数；`validate_config` 在下载前检查关键配置。 |
| `data.py` | 下载并复用原始压缩包；`load_imdb` 完成分层划分、训练词表统计和 CSR 词袋构建；`apply_input_dropout` 实现稀疏输入 Dropout。 |
| `model.py` | `initialize_parameters` 创建零参数，`loss_and_gradients` 实现稳定的逻辑回归损失与解析梯度。 |
| `optimizers.py` | 手写 `Adam`、`AdaGrad`、`RMSProp`、`NesterovSGD`；`create_optimizer` 按配置构造实例。 |
| `training.py` | `train_one_optimizer` 执行小批量训练；`run_optimizer_suite` 保证公平对比，并在所有方法训练结束后评估测试集。 |
| `evaluation.py` | `evaluate` 分批计算 NLL、含可选 L2 的目标与准确率，正确处理末尾小批量权重。 |
| `reporting.py` | `report_results` 打印汇总并保存配置、指标和三联对比图。 |
| `README.md` | 说明任务、公式、运行环境、配置与复现边界。 |

### 2. 默认配置与数据路径

数据来自 Stanford Large Movie Review Dataset 官方 IMDB v1 压缩包，下载地址由 `dataset_url` 配置，原始数据地址为 `https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz`。使用官方 train/test 的有标签影评，每部分各 25,000 条，忽略无标签数据。抽样不减少首次下载体积，完整压缩包约 84 MB。

只有主动运行实验且本地缓存缺失时才下载；先写 `.part` 临时文件，检查压缩包可读取后替换为正式文件。直接读取压缩包成员，不在磁盘解压，也不缓存处理后的特征。缓存损坏会报错，不会自动改用其他来源；这是格式检查，不是密码学完整性验证。

默认 `data_dir="data/imdb"`、`archive_name="aclImdb_v1.tar.gz"`。数据与结果的相对路径均以本实验目录为基准，不受启动工作目录影响；绝对路径也可使用。离线使用时可预先放置来自同一官方来源的压缩包。

默认配置：

- 随机种子 7；训练池分正负类等额抽取 5,000 条，再按类划分 10% 验证集，得到 4,500 条训练、500 条验证；测试集独立等额抽取 2,000 条。
- 分词：解码 HTML 实体、移除标签、转小写，保留英文字母及单词内部的撇号；不去停用词，不做词干化。
- 只在划分后的训练集统计词出现总次数，选择最高频 10,000 词；同频按字母序排序。验证和测试文本不参与词表构建。未知词忽略；若不足 10,000 词，保留 10,000 维并让剩余列为零。
- `binary_bow=True` 表示二值词袋，可设为 `False` 使用词频计数。两种表示都不做长度归一化。
- `dropout_rate=0.5`、`l2_strength=0.0`、批量 128、评估批量 1,024、训练 10 轮。无早停，使用最终参数。
- 各方法共享零初始化，每次训练分别重置 `seed+100` 的排列随机数和 `seed+200` 的 Dropout 随机数。两条随机流互不干扰。
- Adam：基础学习率 `0.01`，`beta1=0.9`，`beta2=0.999`；AdaGrad：基础学习率 `0.1`；RMSProp：基础学习率 `0.01`，`beta2=0.9`；Nesterov：基础学习率 `0.1`，`momentum=0.9`。自适应方法的 `epsilon` 均为 `1e-8`，置于根号外。
- `learning_rate_decay=True` 为四种方法统一启用 $1/\sqrt t$ 衰减。设为 `False` 会改成恒定基础学习率，属于额外实验。

扩大规模可将两个 `max_*_samples` 设为 `None`，将 `epochs` 增至 160，以接近原图横轴范围；仍保留 10% 验证集时实际训练量为 22,500。应仅依据验证集分别调节各方法的学习率及动量参数，不根据测试集选超参数。

扩大数据和训练预算不等于严格复现：分词、二值词袋、训练词表范围、参数化、Dropout 缩放与评估方式未必与原实现一致，当前也没有复现论文的密集网格搜索或多种子统计。若要检查无 Dropout 条件，可将 `dropout_rate` 设为 0 后完整重跑；不要把有无 Dropout 混在同一对比组中。

### 3. main 执行流程

在已配置依赖的 `paper-replication` Conda 环境中，进入实验目录后主动运行：

```bash
conda run -n paper-replication python main.py
```

执行顺序如下：

1. 检查配置；读取或下载 IMDB，抽样并划分验证集，仅从训练集构建词表。
2. 打印实际词表大小、各划分的样本数、正类数、非零密度与空词袋数。
3. 记录 epoch 0 的初始指标，然后依次训练四种方法；逐轮打印训练 NLL、验证 NLL 和验证准确率。发现非有限目标、梯度或参数时终止并报错，不静默跳过异常方法。
4. 所有训练完成后，以最终参数评估测试集；测试指标不参与训练或模型选择。
5. 打印最终训练 NLL、验证准确率、测试 NLL、测试准确率与训练循环累计耗时。耗时不含数据处理、整集评估和绘图。
6. 在 `output_dir` 指定的目录下新建时间戳子目录，保存 `config.json`（配置、数据摘要与运行时依赖版本）、`history.csv`（含 epoch 0 的逐轮指标）、`summary.json`（最终指标）和 `learning_curves.png`（训练 NLL、验证 NLL、验证准确率三联图）。不保存模型、词表或影评文本。
7. 默认保存后显示图像；无图形界面时将 `show_plots=False`，仅保存图像。

观察训练 NLL 在相同 epoch 预算下的下降速度，并用验证曲线检查泛化。如果排序不同于论文，应先检查数据规模、词袋表示和学习率设置，不修改或筛除曲线来迎合预期。导入模块不会触发下载、训练或绘图；交付阶段只做静态核对，未执行实验或测试。

## 五、实验结果

在稀疏特征上，调好参数的 AdaGrad 明显快于 Nesterov SGD，Adam 的收敛速度与 AdaGrad 接近。