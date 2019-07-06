论文： N2N Learning: Network to Network Compression via Policy Gradient Reinforcement Learning 【[pdf](https://arxiv.org/abs/1709.06030)】

作者：[Anubhav Ashok](https://arxiv.org/search/cs?searchtype=author&query=Ashok%2C+A), [Nicholas Rhinehart](https://arxiv.org/search/cs?searchtype=author&query=Rhinehart%2C+N), [Fares Beainy](https://arxiv.org/search/cs?searchtype=author&query=Beainy%2C+F), [Kris M. Kitani](https://arxiv.org/search/cs?searchtype=author&query=Kitani%2C+K+M)

## 摘要

通过强化学习的方式，从teacher模型中"蒸馏"出一个压缩的student模型。

模型中共有两个步骤：

+ 第一步， 采用一个双向LSTM，删除teacher中某些层；

+ 第二步，采用另一个单项LSTM，小心的缩减每个剩余layer的大小。

reword是采用的是子模型的accuracy + compression rate

子模型训练采用knowledge distill



## 一 简介

强化学习方法，从一个larger high-performing model (teacher)中知识蒸馏出一个compressed high-performance architecture (student)。

使用的决策是MDP，马尔科夫决策。

![](https://raw.githubusercontent.com/JiaoYuZhang/picRep/master/img/20190706150356.png)

## 二 相关工作

有两个部分：网络压缩和架构搜索

关于压缩网络的传统方式有：pruning(剪枝)和distillation(蒸馏)。

架构搜索的设计：NAS等。

##三 方法

目标是通过强化学习来学习一种最优的压缩策略(policy)，它以教师网络为输入，系统地将其缩减为一个小的学生网络输出。



### markov desicion process

将整个搜索策略抽象为一个顺序决策的马尔科夫决策过程。

在这个模型中，使用状态$S$来表述一个网络架构。显然，状态$S$的域非常大，因为它包含了teacher网络的每一个可能的简化体系结构。在状态空间$T\left(s^{\prime} | s, a\right)$中，每一个deterministic transition都是通过选择一个action$a$来决定的，比如移除一个卷积过滤器或者减小全连接层的大小。每个action会将一个架构$s$转换成另一个架构$s^{\prime}$。在MDP中，给定特定状态选择action的策略被表述为概率$\pi(a | s)$，这个概率随机的将一个state与一个action相对应。

使用$\mathcal{M}=\{\mathcal{S}, \mathcal{A}, T, r, \gamma\}$来表述这一过程。

![](https://raw.githubusercontent.com/JiaoYuZhang/picRep/master/img/20190706151525.png)

S: state 状态空间，一个有限的集合，由所有可能减少的网络结构组成，可以从教师模型中得到。比如，一个VGG代表一个状态s ∈ S（初始状态），通过移除第一层一个conv filter 可以得到习网络结构s’。

A：actions是可以将一个网络体系结构转换为另一个网络体系结构的有限操作集。在我们的方法中有两类操作类型:层删除操作和层参数减少操作。这些操作的定义将在第3.2.1和3.2.2节中进一步描述。

 T:Transition Function 状态转移方程T : S×A → S。这里，T是确定的，因为一个动作a总是将网络体系结构s转换成最终的网络体系结构s‘的同一概率。

γ是折扣的因素。我们使用γ= 1,这样所有奖励同样有助于最终的回报。

奖励:r: S→r为奖励函数。网络架构r(s)的奖励可以解释为与给定的网络架构s相关联的得分。注意，我们将表示“不完整”网络的中间状态的奖励定义为0，并且仅为最终状态计算一个非平凡的奖励。



###  STUDENT-TEACHER REINFORCEMENT LEARNING

在这种MDP模型下,强化学习的任务是学习最优政策$\pi : \mathcal{S} \rightarrow \mathcal{A}$,它最大化预期的总回报,总给出的奖励:
$$
R(\vec{s})=\sum_{i=0}^{L=|\vec{s}|} r\left(s_{i}\right)=r\left(s_{L}\right)
$$
采用一种策略梯度强化学习方法，根据对奖励的抽样估计迭代更新策略。

为了解决action space过大导致的效率低下，我们提出了一个两阶段强化学习过程。在第一个阶段，策略选择一系列操作，以决定是保留还是删除教师体系结构的每一层。在第二阶段，不同的策略选择一系列离散的操作，这些操作对应于衰减每个剩余层的配置变量的大小。这样，我们就能够有效地探索状态空间，找到最优的学生网络。

![](https://raw.githubusercontent.com/JiaoYuZhang/picRep/master/img/20190706152003.png)

对于删除layer和收缩layer两种策略，我们都对架构进行了反复的采样，并根据架构所获得的奖励来更新策略。

#### layer removal

Stage 1： Layer Removal，操作对应于保留或删除层的二进制决策。

在每个第t步，双向LSTM政策(见图2把隐藏状态$h_{t-1}$,$ h_{t+1}$和当前层信息$x_t$ ： $\pi_{\text { remove }}\left(a_{t} | h_{t-1}, h_{t+1}, x_{t}\right)$。关于当前层l的信息如下:
$$
x_{t}=\left(l, k, s, p, n, s_{\text { start }}, s_{\text { end }}\right)
$$
其中l是层类型、 k内核大小、 s步长、 p填充和n输出数。

$s_{start}$和$s_{end}$用于skip connection。

![](https://raw.githubusercontent.com/JiaoYuZhang/picRep/master/img/20190706152516.png)

#### 

#### layer shrinkage

在layer shrinkage的第t步，关注前一步采样$h_{t-1}$，之前采样的动作， $a_{t-1}$和当前层信息$x_t$：$\pi_{\text { shrink }}\left(a_{t} | a_{t-1}, h_{t-1}, x_{t}\right)$

$x_t$的参数与layer remove相似.

![](https://raw.githubusercontent.com/JiaoYuZhang/picRep/master/img/20190706152825.png)



### Reward Function

本文设计的一个点：希望对高压缩+低精度的模型提供比低压缩+高精度模型更严厉的惩罚。也希望这是一个不依赖于数据集/模型特定超参数的通用奖励函数。所以reward中的值都是比值。

奖励函数定义为：
$$
\begin{aligned} R &=R_{c} \cdot R_{a} \\ &=C(2-C) \cdot \frac{A}{A_{\text { teacher }}} \end{aligned}
$$
其中$C$为学生模型的相对压缩比，$A$为学生模型的验证精度，$A_{teacher}$为教师模型的验证精度，定义为常数。$R_c$和$R_a$分别表示压缩和精度奖励。

压缩奖励$R_c = C(2 - C)$是使用非线性函数计算的，该函数使策略偏向于生成模型，从而在优化压缩的同时保持精度。其中，$C=1-\frac{\# \text { params ( student } )}{\# \text { params ( teacher } )}$。这里需要注意的是，本文使用参数量的比例而不是其他使用量化或编码的压缩方法使用的比特数。

精确度奖励$R_a$的定义为$R_{a}=\frac{A}{A_{\text { teacher }}}$。其中A∈[0,1]表示学生模型的验证精度，$A_{teacher}$表示教师模型的验证精度。我们注意到准确性和压缩奖励都是相对于教师进行标准化的，因此不需要额外的超参数来执行特定于任务的权重。最后，在reward为-1的情况下，策略可能会产生退化(详情原论文附录)。



### Optimization

![](https://raw.githubusercontent.com/JiaoYuZhang/picRep/master/img/20190706154707.png)

此处不赘述，经典强化学习策略梯度算法。



###  KNOWLEDGE DISTILLATION
学生模型使用教师模型标记的数据进行训练。

在knowledge distillation中，不再使用硬标签（hard label），而是使用了教师模型的logits作为label。通过强调教师模型在所有输出中所学到的关系，使学生模型更规范化。

学生模型要做的就是在训练数据上最小化L2 loss$\left\{\left(x^{i}, z^{i}\right)\right\}_{i=1}^{N}$，其中$z_i$是教师模型的logits。Loss函数为：
$$
\mathcal{L}_{\mathrm{KD}}(f(x ; W), z)=\frac{1}{N} \sum_{i}\left\|f\left(x^{(i)} ; W\right)-z^{(i)}\right\|_{2}^{2}
$$
其中$W$为学生网络权重，$f\left(x^{(i)} ; W\right)$为第i个训练数据样本的模型预测。

最后的学生模型的损失函数是综合了hard label和know distillation：
$$
\mathcal{L}(\mathcal{W})=\mathcal{L}_{\text { hard }}\left(f(x ; W), y_{\text { true }}\right)+\lambda * \mathcal{L}_{\mathrm{KD}}(f(x ; W), z)
$$

## 实验结果

![](https://raw.githubusercontent.com/JiaoYuZhang/picRep/master/img/20190706155919.png)

![](https://raw.githubusercontent.com/JiaoYuZhang/picRep/master/img/20190706155946.png)

## 一些思考

1. 怎么去设计约束模型最后计算量的方法

  N2N中，针对的是模型的参数数量做了约束，在reword中加入了参数量比例的“惩罚项”。

  但是，针对于参数量的限制与实际计算速度其实有一定差距，现在的方法更加倾向于直接对运算速度做限制。

2. 如何更好的利用先验知识，来加速/辅助搜索

  第一，在已经设计好的网络上做减法，大大降低了state space的复杂度。

  第二，student model训练时，直接做知识蒸馏学习大网络的权重。

  