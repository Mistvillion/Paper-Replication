# 实验 3：家族树关系补全

## 一、任务理解

复现反向传播论文中的家族树实验：用多层网络学习两棵同构家族树中的亲属关系，并检验网络能否对训练时未见过的关系做出正确补全。

- 输入：一个 (人物, 关系) 查询，如 `(Colin, aunt)`；输出：满足该关系的人物。一个查询可能有多个答案，如 `(Colin, aunt) -> Jennifer, Margaret`，因此输出使用 24 维 multi-hot 表示。
- 数据：24 个人物、12 种关系，两棵同构家族树共产生 112 条事实、104 个查询。`(Colin, mother)`、`(Charlotte, father)`、`(Alfonso, mother)`、`(Sophia, father)` 4 个查询作为留出集，训练只用其余 100 个。
- 评价指标：exact match，即一条查询的 24 个输出（阈值 0.5）全部判对才算正确。目标现象是训练集被拟合，且留出集查询也能被答对。
- 复现范围与简化：只有一个网络、一组默认配置，不做多种子统计。论文给出的内容是任务、数据表示与网络结构 `24→6`、`12→6`、`→12→6→24`。初始化范围、偏置、训练协议和留出划分的细节无法在论文中确认；加权二元交叉熵、Adam、L2 系数、5000 个 epoch、留出的 4 个查询都是本实现的选择，不归因于论文。选用加权 BCE 是因为稀疏的多标签目标在平方误差下容易让输出全部趋近于 0。

## 二、理论实现

网络结构（每层均为 Sigmoid）：

```text
person:   24 -> 6 ┐
                  ├-> 12 -> 6 -> 24
relation: 12 -> 6 ┘
```

符号：批量大小 $B$；人物 one-hot 输入 $X_p\in\mathbb R^{B\times24}$，关系 one-hot 输入 $X_r\in\mathbb R^{B\times12}$；目标 $Y\in\{0,1\}^{B\times24}$。前向传播：

$$
h_p=\sigma(X_pW_p+b_p),\quad h_r=\sigma(X_rW_r+b_r),\quad h_c=\sigma([h_p,h_r]W_c+b_c)
$$

$$
h=\sigma(h_cW_h+b_h),\quad \hat Y=\sigma(hW_o+b_o)
$$

其中 $h_p,h_r\in\mathbb R^{B\times6}$，$h_c\in\mathbb R^{B\times12}$，$h\in\mathbb R^{B\times6}$，$\hat Y\in\mathbb R^{B\times24}$。

训练目标为加权二元交叉熵加 L2 惩罚：

$$
\mathcal L=-\frac1B\sum_i\sum_{j=1}^{24}\left[w\,y_{ij}\log\hat y_{ij}+(1-y_{ij})\log(1-\hat y_{ij})\right]
+\frac\lambda2\sum\|W\|^2
$$

正例权重 $w=3$；L2 只作用于权重，不作用于偏置。报告的“数据损失”是不含 L2 项的第一部分。

输出层误差信号：Sigmoid 与交叉熵组合后化简为 $\delta_o=(\hat Y-Y)\odot w_{ij}/B$，其中 $w_{ij}=w$（$y_{ij}=1$）或 $1$（$y_{ij}=0$）。其余层依次反传：$\delta_{l}=(\delta_{l+1}W_{l+1}^\top)\odot a_l\odot(1-a_l)$。拼接层的误差信号按列拆回，前 6 列属于人物分支，后 6 列属于关系分支。权重梯度为 $\partial\mathcal L/\partial W_l=a_{l-1}^\top\delta_l+\lambda W_l$，偏置梯度为 $\delta_l$ 按样本求和。参数用 Adam 更新（含偏差修正）。

## 三、运行环境

- Python：3.8 或更高版本。
- numpy：1.20 或更高版本，用于数据处理、矩阵运算与手写反向传播。

无需深度学习框架。

## 四、代码说明

### 1. 文件职责

| 文件 | 主要职责 |
| --- | --- |
| `main.py` | 统一入口，串联数据构造、训练、评估与结果打印。 |
| `config.py` | 实验配置。`EXPERIMENT` 管理随机种子、初始化范围、epoch 数、正例权重、L2 系数和演示查询；`ADAM` 管理 Adam 的学习率、`beta1`、`beta2`、`epsilon`。 |
| `data.py` | 人物、关系、婚姻与亲子关系；`build_facts` 推导全部事实，`build_dataset` 编码输入与 multi-hot 目标并划分训练/留出集，`encode_queries` 把任意 (人物, 关系) 查询编码为 one-hot。 |
| `model.py` | 参数初始化 `initialize_parameters`、前向传播 `forward`、含手写反向传播的 `loss_and_gradients`（返回总损失、不含 L2 的数据损失和梯度字典）、手写 `Adam` 类，以及全批量训练函数 `train`（返回训练后的参数和逐 epoch 的数据损失）。 |

exact match 计算、演示查询的预测和打印直接写在 `main.py` 中，不单独拆分模块。

### 2. 默认配置与数据路径

默认值位于 `config.py`：

- 随机种子 `8`；参数在 $[-0.3,0.3]$ 内均匀初始化。
- 全批量训练 `5000` 个 epoch（每个 epoch 更新一次），正例权重 `3.0`，L2 系数 `1e-5`。
- Adam：学习率 `0.01`，`beta1=0.9`，`beta2=0.999`，`epsilon=1e-8`。
- 演示查询 `test_queries` 默认为 `(Colin, mother)`、`(Colin, aunt)`、`(Sophia, father)`，可修改为其他 (人物, 关系)，可用名称见 `data.py` 的 `PEOPLE` 与 `RELATIONS`。留出的 4 个查询由 `data.py` 的 `HOLDOUT_QUERIES` 指定。

数据由 `data.py` 中的家族树定义在内存中直接生成，无需下载，也不读写数据文件。

本实验规模很小，没有需要扩大的样本量；仍存在的差异是：训练协议（加权 BCE、Adam、L2、训练轮数）和初始化范围并非论文给出的设置，留出划分与论文是否一致无法确认；结果依赖随机种子，默认种子是实现者选定的。

### 3. main 执行流程

运行方式：

```bash
python main.py
```

按实际调用顺序：

1. 设置 NumPy 输出精度，调用 `build_dataset` 构造全部查询，划分出 100 条训练查询与 4 条留出查询；该函数内部用断言核对事实、查询与划分的数量。
2. 调用 `train`：按种子初始化参数，仅用训练集做全批量训练，每个 epoch 计算损失与梯度后用 Adam 更新一次。
3. 打印最后一个 epoch 的训练数据损失（不含 L2）。
4. 分别在训练集和留出集上计算 exact match，打印 `训练集 exact match: x/100` 与 `留出集 exact match: x/4`。
5. 对 `test_queries` 中的每个查询打印得分不低于 0.5 的预测人物（括号内为得分）及由家族树推出的已知答案。

据此检查目标现象：训练集 exact match 接近 100/100，留出集的 4 个查询也被答对，演示查询的预测与已知答案一致。本实验不绘图，也不保存文件。

## 五、实验结果

论文展示了网络能在训练中学到可泛化的人物与关系分布式表示，并能补全未见过的关系。本次预期：训练集与留出集的 exact match 都接近满分，演示查询的预测与已知答案一致，但依赖随机种子。

结果未实测：实验代码尚未运行，本文档不给出任何实际指标。
