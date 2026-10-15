# 人工智能经典论文复现

选择标准：论文奠定了今天仍在使用的核心方法；能通过缩小数据、模型或任务做出有意义的教学复现；CV 与 NLP 均覆盖从基础结构到现代 Transformer/检索增强模型的演进。纯历史性文章和综述虽重要，但因不适合“复现”而未列入主线。

> **关于“复现”的边界**：初学阶段不应追求原论文的 ImageNet、WMT、Atari 或 Wikipedia 全量成绩。这里的目标是复现关键机制、对照实验和现象，称为“教学复现”。只有模型、数据、训练预算和评价协议都尽量对齐论文，才是严格复现。

## 一眼看懂学习顺序

### 15 篇必做主线

| 主线步 | 方向 | 论文 | 完成后应掌握 | 仓库起点 |
|---:|---|---|---|---|
| 1 | 通用基础 | Back-propagation | 计算图、链式法则、梯度检查 | `26_cs231n_cnn_fundamentals.ipynb`（相关基础） |
| 2 | 通用基础 | Adam | 一阶优化、动量、二阶矩、偏差修正 | 暂无直接实现 |
| 3 | 通用基础 | Dropout | 随机正则化、训练/推理差异 | `04_rnn_regularization.ipynb`（RNN 扩展） |
| 4 | 通用基础/NLP | LSTM | 门控记忆、BPTT、长依赖 | `03_lstm_understanding.ipynb` |
| 5 | CV | LeNet-5 | 卷积、池化、端到端图像分类 | `26_cs231n_cnn_fundamentals.ipynb`（相关基础） |
| 6 | CV | AlexNet | 深 CNN、ReLU、数据增强、Dropout | `07_alexnet_cnn.ipynb` |
| 7 | CV | ResNet | 残差连接、网络退化、梯度流 | `10_resnet_deep_residual.ipynb` |
| 8 | CV | U-Net | 编码器—解码器、跳连、像素级预测 | 暂无直接实现 |
| 9 | NLP | Neural Probabilistic LM | 词嵌入与神经语言模型 | 暂无直接实现 |
| 10 | NLP | word2vec | CBOW、Skip-gram、负采样 | 暂无直接实现 |
| 11 | NLP | Seq2Seq | 编码器—解码器、教师强制、解码 | `03_lstm_understanding.ipynb`（前置组件） |
| 12 | NLP | Bahdanau Attention | 软对齐、加性注意力、注意力图 | `14_bahdanau_attention.ipynb` |
| 13 | NLP | Transformer | 自注意力、多头注意力、位置编码 | `13_attention_is_all_you_need.ipynb` |
| 14 | NLP | BERT | MLM、双向预训练、下游微调 | `13_attention_is_all_you_need.ipynb`（前置组件） |
| 15 | CV/NLP 交叉 | Vision Transformer | 图像分块、视觉 Token、纯 Transformer 视觉模型 | `13_attention_is_all_you_need.ipynb`（前置组件） |

### 7 篇选做支线

完成主线中的前置项后再插入：

- 生成建模：完成 Back-propagation 与 Adam 后做 **Auto-Encoding Variational Bayes**。
- 强化学习：完成 Adam 与 AlexNet 后做 **DQN**。
- CV 检测/密集预测：完成 ResNet 后做 **Faster R-CNN**；完成 U-Net 前后做 **Dilated Convolutions**。
- 生成式预训练：完成 Transformer 后做 **GPT-1**，再与 BERT 对照。
- 检索增强 NLP：完成 BERT 后依次做 **DPR → RAG**。

如果只走一个方向：

- **CV 最短路线**：Back-propagation → Adam → LeNet-5 → AlexNet → ResNet → U-Net → Transformer → ViT；检测与密集预测再补 Faster R-CNN、Dilated Convolutions。
- **NLP 最短路线**：Back-propagation → Adam → Dropout → LSTM → Neural Probabilistic LM → word2vec → Seq2Seq → Bahdanau Attention → Transformer → BERT；生成式模型插入 GPT-1，知识密集任务再补 DPR、RAG。

## 开始前的最低准备

- 能用 Python/NumPy 写矩阵乘法、Softmax、交叉熵和小批量训练循环。
- 理解导数、链式法则、向量/矩阵形状和训练集/验证集/测试集划分。
- 每次实验固定随机种子，保存配置、损失曲线和最终指标。
- 先写一个最小单元测试，再跑训练；手写算子必须做数值梯度检查。

每篇统一产出四样东西：`README/实验笔记`、`最小实现`、`至少一个对照实验`、`结果图或表`。只抄网络结构而不验证论文声称的现象，不算完成。

---

## 第一阶段：通用人工智能 / 深度学习基础

### 1. Learning representations by back-propagating errors（1986，必做）

- **作者**：David E. Rumelhart、Geoffrey E. Hinton、Ronald J. Williams。
- **核心内容**：用输出误差逐层反传来调整多层网络权重，使隐藏单元自动形成对任务有用的内部表示。
- **主要学习内容**：计算图、链式法则、前向/反向传播、梯度下降、隐藏表示，以及梯度为何能把“责任”分配给前层参数。
- **复现建议**：只用 NumPy 实现两层 MLP，先拟合 XOR，再训练一个小型 MNIST 分类器；对每个参数做有限差分梯度检查。**完成标志**：相对梯度误差小于 `1e-6`，XOR 四个样本全部分类正确，并画出训练损失曲线。
- **仓库起点**：[`26_cs231n_cnn_fundamentals.ipynb`](./26_cs231n_cnn_fundamentals.ipynb) 含神经网络与反向传播相关教学内容，但不是对这篇论文的严格复现。
- **论文链接**：[Nature 原始论文页](https://www.nature.com/articles/323533a0)。

### 2. Adam: A Method for Stochastic Optimization（2015，必做）

- **作者**：Diederik P. Kingma、Jimmy Ba。
- **核心内容**：用梯度的一阶矩与二阶矩的指数移动平均，为每个参数形成自适应更新步长，并在训练早期进行偏差修正。
- **主要学习内容**：SGD、Momentum、RMSProp 与 Adam 的关系；`beta1`、`beta2`、`epsilon` 的作用；一阶矩/二阶矩估计和偏差修正。
- **复现建议**：不要调用框架优化器，手写 SGD、Momentum、RMSProp、Adam，在同一个 MLP 和相同初始化上比较收敛曲线；增加“去掉偏差修正”的消融。**完成标志**：单元测试能手算核对前 3 次参数更新，并给出四种优化器的验证集曲线与结论。
- **仓库起点**：暂无直接实现；可把手写 Adam 接到任意 NumPy Notebook 的训练循环。
- **论文链接**：[arXiv 原始预印本](https://arxiv.org/abs/1412.6980)。

### 3. Long Short-Term Memory（1997，必做）

- **作者**：Sepp Hochreiter、Jürgen Schmidhuber。
- **核心内容**：针对循环网络中误差信号随时间衰减的问题，引入带乘法门控的记忆机制，使信息与梯度能跨更长时间间隔传播。
- **主要学习内容**：普通 RNN 的梯度消失/爆炸、输入/输出门与记忆状态、BPTT、序列状态的形状和缓存。
- **复现建议**：先写单个 LSTM Cell 的前向与反向，再完成 Copy/Adding 任务；比较 Vanilla RNN 与 LSTM 在序列长度增加时的准确率。**完成标志**：LSTM 的梯度检查通过，并在长序列上明显优于同规模普通 RNN。
- **仓库起点**：[`03_lstm_understanding.ipynb`](./03_lstm_understanding.ipynb) 可直接作为教学复现起点。
- **论文链接**：[MIT Press《Neural Computation》原始论文页](https://direct.mit.edu/neco/article/9/8/1735/6109/Long-Short-Term-Memory)。

### 4. Dropout: A Simple Way to Prevent Neural Networks from Overfitting（2014，必做）

- **作者**：Nitish Srivastava、Geoffrey Hinton、Alex Krizhevsky、Ilya Sutskever、Ruslan Salakhutdinov。
- **核心内容**：训练时随机丢弃神经元及其连接，减少特征之间的共适应；推理时使用完整网络近似许多稀疏子网络的模型平均。
- **主要学习内容**：过拟合、Bernoulli Mask、训练/推理阶段差异、期望缩放、随机正则化与近似模型集成。
- **复现建议**：在故意缩小的 Fashion-MNIST 训练集上，用相同 MLP 比较 `dropout=0/0.2/0.5`；固定数据划分和训练步数。**完成标志**：报告训练—验证间隙并画曲线，说明为何训练准确率下降不代表正则化失败。
- **仓库起点**：[`04_rnn_regularization.ipynb`](./04_rnn_regularization.ipynb) 对应后续 *Recurrent Neural Network Regularization*，可学习 Dropout 在序列模型中的放置；它不是本篇论文的严格复现。
- **论文链接**：[JMLR 原始论文页](https://www.jmlr.org/papers/v15/srivastava14a.html)。序列模型扩展的一手来源见 [Recurrent Neural Network Regularization](https://arxiv.org/abs/1409.2329)。

### 5. Auto-Encoding Variational Bayes（2014，选做）

- **作者**：Diederik P. Kingma、Max Welling。
- **核心内容**：通过可重参数化的变分下界估计器，把含连续潜变量的近似后验学习转化为可用随机梯度优化的训练问题，并用识别模型摊销推断成本。
- **主要学习内容**：生成模型、近似后验、ELBO、KL 散度、重参数化技巧、重建质量与潜空间正则之间的权衡。
- **复现建议**：在 MNIST 或二维合成数据上实现小型 VAE，绘制重建、先验采样、潜空间插值和 ELBO 两部分曲线；比较去掉 KL 项的普通自编码器。**完成标志**：能解释并分别记录重建项与 KL 项，采样和插值结果合理。
- **仓库起点**：[`17_variational_autoencoder.ipynb`](./17_variational_autoencoder.ipynb) 实现了 VAE 核心机制，但其标注对应的是后续论文 *Variational Lossy Autoencoder*，因此应作为概念热身而非原论文严格复现。
- **论文链接**：[arXiv 原始预印本](https://arxiv.org/abs/1312.6114)。

### 6. Human-level control through deep reinforcement learning（2015，选做）

- **作者**：Volodymyr Mnih、Koray Kavukcuoglu、David Silver、Andrei A. Rusu、Joel Veness、Marc G. Bellemare、Alex Graves、Martin Riedmiller、Andreas K. Fidjeland、Georg Ostrovski、Stig Petersen、Charles Beattie、Amir Sadik、Ioannis Antonoglou、Helen King、Dharshan Kumaran、Daan Wierstra、Shane Legg、Demis Hassabis。
- **核心内容**：将卷积表示学习与 Q-learning 结合为 DQN，直接从像素和奖励学习控制策略；经验回放与目标网络是稳定训练的关键组件。
- **主要学习内容**：MDP、Q-learning、Bellman 目标、探索与利用、经验回放、目标网络、强化学习评估的高方差。
- **复现建议**：不要从 Atari 全量开始；先在 CartPole 或简化 GridWorld 实现 DQN，再选单个 Atari 环境做缩小实验。至少比较“有/无经验回放”和“有/无目标网络”。**完成标志**：用不少于 5 个随机种子报告平均回报与方差，而不是只展示最好的一次。
- **仓库起点**：暂无直接实现；AlexNet Notebook 只能提供卷积前端的相关经验。
- **论文链接**：[Nature 原始论文页](https://www.nature.com/articles/nature14236)。

---

## 第二阶段：计算机视觉（CV）

### 7. Gradient-Based Learning Applied to Document Recognition（1998，必做；LeNet-5）

- **作者**：Yann LeCun、Léon Bottou、Yoshua Bengio、Patrick Haffner。
- **核心内容**：把卷积网络、梯度学习与文档识别任务结合成端到端系统；LeNet-5 展示了局部连接、权重共享和下采样对视觉任务的价值。
- **主要学习内容**：二维卷积、感受野、权重共享、池化/下采样、Softmax 类别预测，以及从像素到类别的端到端训练。
- **复现建议**：在 MNIST 上实现接近 LeNet-5 的小网络；第一版只用 NumPy 写卷积前向与反向，第二版可用框架验证。**完成标志**：卷积梯度检查通过，测试准确率稳定，并可视化第一层卷积核和特征图。
- **仓库起点**：[`26_cs231n_cnn_fundamentals.ipynb`](./26_cs231n_cnn_fundamentals.ipynb) 适合作为 CNN 基础热身，但不是 LeNet-5 原论文复现。
- **论文链接**：[IEEE Xplore 原始论文页](https://ieeexplore.ieee.org/document/726791)。

### 8. ImageNet Classification with Deep Convolutional Neural Networks（2012，必做；AlexNet）

- **作者**：Alex Krizhevsky、Ilya Sutskever、Geoffrey E. Hinton。
- **核心内容**：在大规模 ImageNet 分类上训练深卷积网络，结合 ReLU、GPU 计算、数据增强与 Dropout，大幅推进当时的图像分类结果。
- **主要学习内容**：深 CNN 的层次特征、ReLU 与饱和激活、最大池化、局部响应归一化的历史位置、数据增强、Dropout 和 Top-k 指标。
- **复现建议**：不要训练原版 ImageNet；在 CIFAR-10 或 Tiny ImageNet 上缩小通道数与全连接层，保留“卷积—ReLU—池化—全连接”骨架。做 ReLU 对 Sigmoid、带/不带增强的对照。**完成标志**：报告参数量、训练时间、Top-1 准确率和两项消融曲线。
- **仓库起点**：[`07_alexnet_cnn.ipynb`](./07_alexnet_cnn.ipynb) 可直接作为教学复现起点。
- **论文链接**：[NeurIPS 原始论文页](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)。

### 9. Deep Residual Learning for Image Recognition（2016，必做；ResNet）

- **作者**：Kaiming He、Xiangyu Zhang、Shaoqing Ren、Jian Sun。
- **核心内容**：让堆叠层学习相对输入的残差而不是直接学习未参照映射，缓解深网络的优化与退化问题，使网络能可靠地加深。
- **主要学习内容**：恒等快捷连接、BasicBlock/Bottleneck、维度变化时的投影、网络退化与过拟合的区别、梯度流。
- **复现建议**：在 CIFAR-10 上实现 ResNet-20，并训练参数量相近的 PlainNet；记录训练误差、验证误差与早期层梯度范数。**完成标志**：复现“更深 PlainNet 不一定更好训练，而残差网络更易优化”的方向性现象。
- **仓库起点**：[`10_resnet_deep_residual.ipynb`](./10_resnet_deep_residual.ipynb) 可直接开始；完成后可用 [`15_identity_mappings_resnet.ipynb`](./15_identity_mappings_resnet.ipynb) 深入预激活残差结构。
- **论文链接**：[CVPR Open Access 原始论文页](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)。

### 10. U-Net: Convolutional Networks for Biomedical Image Segmentation（2015，必做）

- **作者**：Olaf Ronneberger、Philipp Fischer、Thomas Brox。
- **核心内容**：采用对称的收缩路径与扩张路径，并把高分辨率编码特征通过跳跃连接交给解码器，在少量标注和强数据增强下完成精确分割。
- **主要学习内容**：语义分割、编码器—解码器、多尺度特征、裁剪/拼接跳连、像素级损失、IoU 与 Dice。
- **复现建议**：先用合成几何图形分割验证全流程，再用小型细胞或 Oxford-IIIT Pet 子集；比较有/无跳连和有/无增强。**完成标志**：输出掩码可视化、IoU/Dice、两项消融，并能解释跳连为何改善边界。
- **仓库起点**：暂无直接实现；可复用 AlexNet/ResNet Notebook 的卷积基础。
- **论文链接**：[作者实验室原始论文与代码页](https://lmb.informatik.uni-freiburg.de/Publications/2015/RFB15a/)。

### 11. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks（2015，选做）

- **作者**：Shaoqing Ren、Kaiming He、Ross Girshick、Jian Sun。
- **核心内容**：用与检测器共享卷积特征的 Region Proposal Network 学习候选区域，使候选生成接近零额外开销，并将检测流程进一步端到端化。
- **主要学习内容**：Anchor、objectness、边界框回归、IoU、正负样本匹配、NMS、RoI 特征与 mAP。
- **复现建议**：先在合成矩形或 Penn-Fudan 小数据集上做单类别检测；第一阶段只验证 RPN 的 proposal recall，再接分类/回归头。**完成标志**：可视化 Anchor、proposal 和最终框，报告 Recall@IoU 与 AP；不要一开始挑战 COCO。
- **仓库起点**：暂无直接实现；先完成 ResNet 作为 Backbone 基础。
- **论文链接**：[NeurIPS 原始论文页](https://papers.nips.cc/paper/2015/hash/14bfa6bb14875e45bba028a21ed38046-Abstract.html)。

### 12. Multi-Scale Context Aggregation by Dilated Convolutions（2016，选做）

- **作者**：Fisher Yu、Vladlen Koltun。
- **核心内容**：用空洞卷积在不降低特征图分辨率的情况下指数扩大感受野，从而为密集预测系统性地聚合多尺度上下文。
- **主要学习内容**：有效感受野、dilation rate、密集预测、分辨率与上下文范围的权衡、多尺度上下文模块。
- **复现建议**：先手写二维空洞卷积并可视化采样位置，再在合成分割数据上比较标准卷积、池化扩大感受野和空洞卷积。**完成标志**：形状/数值单元测试通过，画出各方案的感受野，并报告相同或近似参数量下的 IoU。
- **仓库起点**：[`11_dilated_convolutions.ipynb`](./11_dilated_convolutions.ipynb) 是直接的教学实现。
- **论文链接**：[arXiv 原始预印本](https://arxiv.org/abs/1511.07122)。

### 13. An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale（2021，必做；ViT）

- **作者**：Alexey Dosovitskiy、Lucas Beyer、Alexander Kolesnikov、Dirk Weissenborn、Xiaohua Zhai、Thomas Unterthiner、Mostafa Dehghani、Matthias Minderer、Georg Heigold、Sylvain Gelly、Jakob Uszkoreit、Neil Houlsby。
- **核心内容**：把图像切为 Patch 序列并线性嵌入为 Token，使用接近标准 Transformer Encoder 的纯注意力模型进行图像分类，说明卷积并非视觉识别的必要组件。
- **主要学习内容**：Patch Embedding、`[CLS]` Token、位置嵌入、Encoder Block、归纳偏置、预训练规模与迁移学习。
- **复现建议**：先完成本路线 Transformer，再在 CIFAR-10 上实现 Tiny ViT（如 patch=4、6 层 Encoder）；与参数量接近的小 CNN 比较，并改变 patch 大小。**完成标志**：画出准确率/训练速度对照和 patch-size 消融，解释小数据上 ViT 可能不占优的原因。
- **仓库起点**：暂无 ViT 直接实现；[`13_attention_is_all_you_need.ipynb`](./13_attention_is_all_you_need.ipynb) 提供核心注意力组件。
- **论文链接**：[ICLR OpenReview 原始论文](https://openreview.net/forum?id=YicbFdNTTy)。

---

## 第三阶段：自然语言处理（NLP）

### 14. A Neural Probabilistic Language Model（2003，必做）

- **作者**：Yoshua Bengio、Réjean Ducharme、Pascal Vincent、Christian Jauvin。
- **核心内容**：同时学习词的分布式表示与词序列概率函数，使相似词共享统计强度，从而缓解离散序列组合爆炸带来的维数灾难。
- **主要学习内容**：n-gram 语言模型、词嵌入、固定窗口上下文、条件概率、交叉熵与困惑度、分布式表示。
- **复现建议**：在小型语料上先写带平滑的 n-gram 基线，再实现固定窗口前馈神经语言模型；可视化词向量邻居。**完成标志**：正确计算验证集 perplexity，并比较不同上下文长度和 n-gram 基线。
- **仓库起点**：暂无直接实现；[`02_char_rnn_karpathy.ipynb`](./02_char_rnn_karpathy.ipynb) 可作为语言建模训练循环的后续参照，但模型结构不同。
- **论文链接**：[JMLR 原始论文页](https://jmlr.org/papers/v3/bengio03a.html)。

### 15. Efficient Estimation of Word Representations in Vector Space（2013，必做；word2vec）

- **作者**：Tomas Mikolov、Kai Chen、Greg Corrado、Jeffrey Dean。
- **核心内容**：提出计算高效的 CBOW 与 Skip-gram 架构，从大规模文本学习连续词向量，并用词法和语义相似关系评价表示质量。
- **主要学习内容**：上下文窗口、CBOW、Skip-gram、Hierarchical Softmax，以及词向量相似度与类比评估。
- **复现建议**：实现 Skip-gram；为便于复现高效训练，可结合后续论文提出的 Negative Sampling，在小型语料或 Text8 子集上训练。做窗口大小、负样本数与词频下采样的消融。**完成标志**：最近邻结果具有可解释性，并报告相似度/类比或自建可控测试集结果。
- **仓库起点**：暂无直接实现。
- **论文链接**：[arXiv 原始预印本](https://arxiv.org/abs/1301.3781)。补充 Negative Sampling 的一手来源见 [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)。

### 16. Sequence to Sequence Learning with Neural Networks（2014，必做）

- **作者**：Ilya Sutskever、Oriol Vinyals、Quoc V. Le。
- **核心内容**：用多层 LSTM 编码器把输入序列压缩为固定维向量，再用 LSTM 解码器生成目标序列，形成通用端到端序列映射方法；论文还展示了反转源序列对优化的帮助。
- **主要学习内容**：Encoder–Decoder、教师强制、`<BOS>/<EOS>`、贪心/Beam Search、序列掩码、BLEU，以及固定向量瓶颈。
- **复现建议**：先做 Copy/Reverse/日期格式转换，再做小型英法或中英数据；严格比较源序列正常顺序与反转顺序。**完成标志**：训练时与自回归推理都能工作，并报告序列级准确率或 BLEU 与顺序对照。
- **仓库起点**：[`03_lstm_understanding.ipynb`](./03_lstm_understanding.ipynb) 提供 LSTM 前置知识，但需自行补 Encoder–Decoder。
- **论文链接**：[NeurIPS 原始论文页](https://papers.nips.cc/paper_files/paper/2014/hash/5a18e133cbf9f257297f410bb7eca942-Abstract.html)。

### 17. Neural Machine Translation by Jointly Learning to Align and Translate（2015，必做；Bahdanau Attention）

- **作者**：Dzmitry Bahdanau、Kyunghyun Cho、Yoshua Bengio。
- **核心内容**：不再强迫编码器把整句压进单个固定向量，而是在生成每个目标词时，对源序列位置计算可学习的软对齐并汇聚相关上下文。
- **主要学习内容**：加性注意力、Query/Key/Value 的前身直觉、对齐分数、注意力权重、上下文向量和注意力可视化。
- **复现建议**：沿用上一项同一数据与 LSTM 基线，加入加性注意力；按句长分桶比较无注意力与有注意力模型，并画对齐热图。**完成标志**：注意力权重归一化测试通过，在较长序列上优于固定向量基线，热图可解释。
- **仓库起点**：[`14_bahdanau_attention.ipynb`](./14_bahdanau_attention.ipynb) 可直接作为教学复现起点。
- **论文链接**：[arXiv 原始预印本（ICLR 论文）](https://arxiv.org/abs/1409.0473)。

### 18. Attention Is All You Need（2017，必做；Transformer）

- **作者**：Ashish Vaswani、Noam Shazeer、Niki Parmar、Jakob Uszkoreit、Llion Jones、Aidan N. Gomez、Łukasz Kaiser、Illia Polosukhin。
- **核心内容**：以多头自注意力和逐位置前馈网络构成 Encoder–Decoder，完全去掉循环与卷积，使序列训练更易并行，并缩短长距离依赖的计算路径。
- **主要学习内容**：Scaled Dot-Product Attention、多头注意力、位置编码、残差与 LayerNorm、Padding/Causal Mask、标签平滑、Warmup 学习率和自回归解码。
- **复现建议**：按顺序实现并测试 mask → 单头注意力 → 多头注意力 → Encoder Block → Decoder；在 Copy 或小型翻译任务训练 Tiny Transformer。**完成标志**：未来位置的注意力权重严格为 0，所有张量形状测试通过，模型能自回归完成任务，并可视化至少一个注意力头。
- **仓库起点**：[`13_attention_is_all_you_need.ipynb`](./13_attention_is_all_you_need.ipynb) 是最直接的教学实现。
- **论文链接**：[NeurIPS 原始论文页](https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)。

### 19. Improving Language Understanding by Generative Pre-Training（2018，选做；GPT-1）

- **作者**：Alec Radford、Karthik Narasimhan、Tim Salimans、Ilya Sutskever。
- **核心内容**：先在大量无标注文本上训练自回归 Transformer 语言模型，再用很少的任务结构改动对不同理解任务进行监督微调，展示“预训练 → 微调”的通用迁移范式。
- **主要学习内容**：Decoder-only Transformer、Causal LM、无监督预训练、任务输入变换、迁移学习与零初始化辅助目标。
- **复现建议**：训练 2–6 层 Tiny GPT 做字符或子词级因果语言建模，再把预训练权重微调到情感分类；对照随机初始化分类器。**完成标志**：报告预训练 perplexity、生成样例及有/无预训练的少样本分类曲线；不追求原论文规模。
- **仓库起点**：[`13_attention_is_all_you_need.ipynb`](./13_attention_is_all_you_need.ipynb) 提供注意力组件，[`02_char_rnn_karpathy.ipynb`](./02_char_rnn_karpathy.ipynb) 提供自回归文本生成对照。
- **论文链接**：[OpenAI 原始论文 PDF](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)。

### 20. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding（2019，必做）

- **作者**：Jacob Devlin、Ming-Wei Chang、Kenton Lee、Kristina Toutanova。
- **核心内容**：用 Masked Language Model 学习深层双向 Transformer 表示，并通过增加很小的任务输出层对多类 NLP 任务进行统一微调；原论文还使用 Next Sentence Prediction。
- **主要学习内容**：Transformer Encoder、MLM、NSP、`[CLS]/[SEP]/[MASK]`、Segment Embedding、预训练与微调的数据流，以及双向编码与因果解码的区别。
- **复现建议**：在小语料上训练 2–4 层 Mini-BERT，至少实现 MLM，再微调文本分类；若实现 NSP，单独做消融，不把它与 MLM 混成一个结论。**完成标志**：MLM 准确率显著高于随机基线，有预训练的分类模型在少标注设置下优于从零训练。
- **仓库起点**：暂无 BERT 直接实现；[`13_attention_is_all_you_need.ipynb`](./13_attention_is_all_you_need.ipynb) 可提供 Encoder 注意力基础。
- **论文链接**：[ACL Anthology / NAACL 原始论文页](https://aclanthology.org/N19-1423/)。

### 21. Dense Passage Retrieval for Open-Domain Question Answering（2020，选做；DPR）

- **作者**：Vladimir Karpukhin、Barlas Oğuz、Sewon Min、Patrick Lewis、Ledell Wu、Sergey Edunov、Danqi Chen、Wen-tau Yih。
- **核心内容**：用问题编码器与段落编码器把两者映射到同一稠密向量空间，通过内积和批内负样本训练可扩展检索器，在开放域问答中替代或补充稀疏检索。
- **主要学习内容**：Bi-encoder、对比学习、批内负样本、hard negative、最大内积搜索、Recall@k，以及检索器与阅读器的职责边界。
- **复现建议**：先用几十到几千条自建“问题—支持段落”数据训练小型双编码器，暴力内积检索即可；比较随机负样本与批内负样本。**完成标志**：报告 Recall@1/5/20，并检查同一问题的正确段落得分高于困难负样本。
- **仓库起点**：[`28_dense_passage_retrieval.ipynb`](./28_dense_passage_retrieval.ipynb) 是直接的教学实现。
- **论文链接**：[ACL Anthology / EMNLP 原始论文页](https://aclanthology.org/2020.emnlp-main.550/)。

### 22. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks（2020，选做；RAG）

- **作者**：Patrick Lewis、Ethan Perez、Aleksandra Piktus、Fabio Petroni、Vladimir Karpukhin、Naman Goyal、Heinrich Küttler、Mike Lewis、Wen-tau Yih、Tim Rocktäschel、Sebastian Riedel、Douwe Kiela。
- **核心内容**：把预训练 Seq2Seq 生成器的参数记忆与外部稠密向量索引的非参数记忆结合，提出对整段输出共享文档的 RAG-Sequence 和可逐 Token 改变文档的 RAG-Token。
- **主要学习内容**：检索—生成流水线、潜文档边缘化、参数/非参数记忆、证据来源、索引更新，以及检索召回率对生成质量的上限约束。
- **复现建议**：接着 DPR 的小语料，用 Tiny Seq2Seq 或现成小模型完成“检索后生成”；先实现简化的 RAG-Sequence，再比较无检索、随机文档、正确检索三组。**完成标志**：同时报告 Recall@k 和答案 EM/F1，保存答案所用文档，能区分“没检索到”与“检索到但生成错”。
- **仓库起点**：先完成 [`28_dense_passage_retrieval.ipynb`](./28_dense_passage_retrieval.ipynb)，再进入 [`29_rag.ipynb`](./29_rag.ipynb)。
- **论文链接**：[NeurIPS 原始论文页](https://papers.nips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html)。

---

## 推荐的单篇复现工作流

1. **先读摘要、图 1、方法、实验表**：写下论文要解决的问题、关键改动、作者用什么实验支持结论。成功标志：能用 3 句话讲清“旧方法哪里不够、论文改了什么、证据是什么”。
2. **先画张量形状，再写代码**：把每层输入/输出和损失函数写在纸面或 Markdown。成功标志：所有形状在 batch 与序列/空间维度上自洽。
3. **从最小数据过拟合开始**：只用 8–32 个样本训练。成功标志：模型能把训练损失压到很低；做不到时先排查实现，不急着加数据。
4. **复现一个核心现象**：例如 ResNet 对 PlainNet、注意力对固定向量、DPR 的批内负样本对随机负样本。成功标志：有同设置的对照表，而非只有单个最终数字。
5. **再做缩小版正式实验**：固定种子和配置，保存日志、曲线、模型大小与运行时间。成功标志：另一个人能按记录重复运行。
6. **最后写偏差说明**：列出与原论文在数据、模型、预算、评价上的差异。成功标志：不把教学复现的数字冒充原论文复现结果。

## 时间与算力建议

- **只有 CPU**：优先完成 1–5、9–12、14–18；每个实验把数据控制在几千到几万样本，先验证机制。
- **单张消费级 GPU**：可以完成全部必做，并做 Tiny ViT、Mini-BERT、单类别 Faster R-CNN 和小语料 DPR/RAG。
- **不建议初学者直接做**：ImageNet 原版 AlexNet/ResNet/ViT、WMT 原版翻译、完整 BERT 预训练、49 个 Atari、全 Wikipedia RAG。它们主要考验算力与工程，不利于第一次理解算法。

## 建议的阶段验收

- **基础阶段**：能手写反向传播与 Adam；能解释 LSTM 的时间梯度通路；知道正则化对训练/验证误差的不同影响。
- **CV 阶段**：能从零实现卷积与残差块；能区分分类、检测、分割的输出和指标；能把 Transformer 改造成 ViT。
- **NLP 阶段**：能从窗口语言模型走到 LSTM Seq2Seq、注意力、Transformer、BERT；能区分自回归与掩码语言模型。
- **高级阶段**：能将检索误差与生成误差分开评估，并为每个实验提供可复查的对照和偏差说明。

## 来源说明

本文的题名、作者与方法概要均依据论文原始页面、出版社/会议论文集、ACL Anthology、OpenReview 或作者发布的 arXiv/OpenAI 原文核对。每个条目的“论文链接”均指向一手来源；复现规模与验收指标是为本仓库初学者路线做出的教学建议，并非原论文声称的实验设置。
