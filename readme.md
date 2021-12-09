# 强化学习

## Qlearning

更新一个 Q 表，表中的每个元素代表每个状态下每个动作的潜在奖励<br>
根据 Q 表选择动作，然后更新 Q 表

```
state 1 2 3 4 5
left  0 0 0 0 0
right 0 0 0 1 0
```

更新策略：`现实值=现实值+lr*（估计值-现实值）`

---
## Sarsa

Qlearning 更新方法：`根据当前Q表选择动作->执行动作->更新Q表`<br>
Sarsa 更新方法：`执行动作->根据当前估计值选择下一步动作->更新Q表`

**Sarsa 是行动派，Qlearning 是保守派**

---
## SarsaLambda

Sarsa 的升级版<br>
Qlearning 和 Sarsa 都认为上一步对于成功是有关系的，但是上上一步就没有关系了，SarsaLambda 的思想是：`到达成功的每一步都是有关系的，他们的关系程度为：越靠近成功的步骤是越重要的`<br>

```
step
1-2-3-4-5-success
重要性1<2<3<4<5
```
---
## DQN
![](./DeepQLearningNetwork/dqn.jpg)<br>
用神经网络代替 Q 表的功能

Q 表无法进行所有情况的枚举，在某些情况下是不可行的，比如下围棋。<br>
Features: `Expericence Replay and Fixed Q-targets`

Experience Replay : `将每一次实验得到的惊艳片段记录下来，然后作为经验，投入到经验池中，每次训练的时候随机取出一个 BATCH，可以复用数据。`

Fixed Q-target: `在神经网络中，Q 的值并不是互相独立的，所以不能够进行分别更新操作，那么我们需要将网络参数复制一份，解决该问题。`

为了解决 overestimate 的问题，引入 double DQN，算法上有一点点的改进，复制一份网络参数，两个网络的参数异步更新

---
## DQN with Prioritized Experience Replay
在DQN中，我们有Experience Replay，但是这是经验是随机抽取的，我们需要让好的、成功的记忆多多被学习到，所以我们在抽取经验的时候，就需要把这些记忆优先给网络学习，于是就有了`Prioritized`Experience Replay

## Dueling DQN
将Q值的计算分成状态值state_value和每个动作的值advantage，可以获得更好的性能

---
## Policy Gradient

核心思想：让好的行为多被选择，坏的行为少被选择。<br>
采用一个参数 vt，让好的行为权重更大<br>
![](./PolicyGradient/5-1-1.png)<br>

---
## ActorCritic

使用神经网络来生成 vt，瞎子背着瘸子

---

## DDPG
![](DeepDeterministicPolicyGradient\principle.png)
## Requirements

- torch
- gym
- tensorboardX

---
靠，pytorch 官网上有：

https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
