# 强化学习

## 目录

[TOC]

## 写在前面

- Note that the algorithm code comes from some experts in the field of reinforcement learning or I refactored the algorithms myself.

- 本仓库中的强化学习算法来自于Medium、YouTube、CSDN等等网站，详细的信息请见该readme下面的“参考资料”这一小节，或许会对您有些帮助。

- 此外，`TRPO`我似乎没有搞懂，代码并没有调试，请不要运行TRPO的代码。

- 有些公式无法显示不知道为什么，`git clone 到本地使用VSCODE能够完整显示`，代码基本上都经过运行调试并能够生成tensorboard记录，如果遇到bug，请给我提issue，O(∩_∩)O。

- 最近在搞MARL，后期会添加一些MARL的算法。

## 进度

where the \* mark means the algorithm is important and worth diving into it

|         method         | done |
| :-------------------- | :----: |
|      \*Qlearning       | √    |
|         Sarsa          | √    |
|      SarsaLambda       | √    |
|         \*DQN          | √    |
|      \*DQNwithPER      | √    |
|       DuelingDQN       | √    |
|   \*Policy Gradient    | √    |
|      \*AC and A2C      | √    |
|          ACER          | √    |
|          A3C           | √    |
|  \*SAC (PER optional)  | √    |
|         \*DDPG         | √    |
| TD3 (PER,HER optional) | √    |
|          \*TRPO        | √    |
|         \*PPO          | √    |
|          DPPO          | √    |
|          Multi-Agent DDPG          | √    |
|          Multi-Agent PPO          | ×     |
|         DIAYN          | ×    |

---

## 术语表

仿造[这个博客](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)也造了个术语表,这些术语都会在下面看到。

| 符号| 意义|
|:--:|----|
|$s \in \mathcal{S}$| 状态|
|$a \in \mathcal{A}$| 动作|
|$r \in \mathcal{R}$| 奖励|
|$S_t,A_t,R_t$|一个轨迹中时间$t$时候的状态、动作和奖励，其中大写的是随机变量，小写的是观测值|
|$U_t$ | 被称为Return或者是未来的折扣奖励(discounted rewards),有的文章也叫 $G_t$ ，总之意义一样，$U_t=\gamma^0 R_t+\gamma^1R_{t+1}+\gamma^2R_{t+2}+...$，其中 $\gamma$ 为discount factor折扣因子|
| $P(s',r\vert s,a)$|从当前状态 $s$ 采取动作 $a$，转移到 $s_{t+1}$ 状态并获取奖励$r$的概率|
|$\pi(a\vert s)$|随机策略,$\pi(a \vert s; \theta)$ 表示由 $\theta$ 作为参数初始化的策略网络|
|$\mu(s)$|确定性的策略，在DDPG中可以见到，也可以表示为$\pi(s)$，一般用前者|
|$V(s)$| 状态价值函数，它预测状态$s$下的期望折扣奖励，其中$V(s;\mathbf{w})$是由参数$\mathbf{w}$初始化的价值网络|
| $V_\pi(s)$| 当我们遵循策略$\pi$的时候，状态$s$的价值，其中$ V_\pi(s) = \mathbb{E}_{a \sim \pi(\cdot \vert s)} [U_t \vert S_t =s]$ |
|$Q(s,a)$|在状态$s$时采取动作$a$的时候能够的到的期望折扣奖励(return)，其中$Q(s,a;\mathbf{w})$代表以参数$\mathbf{w}$初始化的动作价值函数|
|$Q_\pi(s,a)$|遵循策略$\pi$的时候，在状态$s$时采取动作$a$的时候能够的到的期望折扣奖励(return)，$Q(s,a;\mathbf{w}) = \mathbb{E}_{a \sim \pi(\cdot \vert s)} [U_t \vert S_t =s,A_t=a]$|
| $A(s,a)$| 优势函数(advantage)，$A(s,a) = Q(s,a) - V(s)$ 它是另外一种Q-value，加入 $V(s)$ 作为baseline，具有更低的方差|
|$r(\pi)$|根据策略$\pi$选取动作，能够得到的**平均回报**|
|$d^\pi(s)$|策略$\pi$所决定的马尔科夫链的平稳分布|

## 1. 关键概念 Key Concepts

1. **代理(agent)在一个环境(environment)中执行动作/行为(action)。环境如何对代理的动作做出响应由一个已知或未知的模型(model)来定义。执行代理可以停留在环境中的某个状态(state) $s\in \mathcal{S}$，可以通过执行某个行为/动作(action) $a\in \mathcal{A}$来从一个状态$s$进入到另一个状态$s'$。代理会到达什么状态由状态转移概率$(P)$决定。代理执行了一个动作之后，环境会给出一定的奖励(reward) $r\in\mathcal{R}$作为反馈。**

2. 几乎所有的强化学习问题可以使用马尔科夫决策过程（MDPs）来描述，MDP 中的所有状态都具有“马尔科夫性”：未来仅仅依赖于当前的状态，并不与历史状态相关，在给定当前状态下，未来与过去条件独立，也就是**当前状态包含了决定未来所需的所有信息**。

3. 策略：即智能体 agent 的行为函数 $\pi$，是当前状态到一个动作的映射，它可以是随机性的(random)也可以是确定性的(deterministic)：

   1. $\pi(s)=a$
   2. $\pi(a \mid s)= \mathbb{P}_{\pi}(A=a \mid S=s)$

4. 动作-价值函数(Action-value Function) $Q(s,a)$：动作-价值函数是衡量一个状态或者是一个`(状态，行为)元组`的好坏，它是 $U_t$ 的期望:$Q_{\pi}(s_t,a_t) = \mathbb{E}[U_t\vert S_t = s_t, A_t = a_t]$；未来的奖励（称为`回报`）定义为带衰减的后续奖励之和(discounted rewards)

   1. $$ U_t = R_{t+1} + \gamma R_{t+2} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} $$
      在[下面第六节](#1-关键概念-key-concepts)中会详细讲到

   2. $\gamma$ 作为对未来奖励的`惩罚`(`penaty`)，因为：
      1. 未来奖励的不确定性
      2. 未来奖励不会直接提供收益
      3. 数学上便利，无需在乎太远的奖励，被 $\gamma$ 衰减掉了
      4. 使用衰减系数，无需担心存在无限循环的转移图
   3. $Q^*(s_t,a_t)=\mathop{\max}_{\pi} Q_{\pi}(s_t,a_t)$，可以对 $a_t$ 做评价，这个动作有多好,求出来了$Q^*$之后，Agent便可以根据这个动作价值函数选取最优的动作

5. 状态-价值函数(State-Value Function)存在两种形式：状态 $s$ 的状态价值——`回报的期望值`；某个（state，action）元组的行为价值函数——`该行为相比于平均状态能够获得多大收益`？

    $V(s)$能够表示当前局势的好坏。而$Q_{\pi}(s,a)$能够衡量Agent在s状态下选取动作a的好坏。

   1. 我们可以利用行为的分布以及行为的价值函数来推导`状态价值函数`
      $$ \begin{aligned}V_{\pi}(s) &= \sum_{a \in \mathcal{A}} Q_{\pi}(s, a) \pi(a \vert s) \\&=\mathbb{E}_A[Q_{\pi}(s, A)]
      \end{aligned}$$
   2. 定义行为价值函数和状态价值函数之间的差称为`优势(advantage)`函数，意味着这个动作比`平均状态`好多少？
      $$ A_{\pi}(s, a) = Q_{\pi}(s, a) - V_{\pi}(s) $$
   3. 对$V_{\pi}(S)$求期望$\mathbb{E}_S[V_{\pi}(S)]$，我们可以得到这个 policy $\pi$ 的好坏

6. 贝斯曼方程与 Return(aka cumulative future reward),**注意Return跟上面的奖励Reward是不一样的。**

   1. 贝尔曼方程指的是一系列的等式，它将价值函数分解为直接奖励加上衰减后的未来奖励。(discounted rewards)
   2. return(aka cumulative future reward), `Return并不是reward`，它可以这么表示: $U_t={R_t}+R_{t+1}+R_{t+2}+...$
   3. discounted return (aka cumulative discounted future reward) : $U_t=\gamma^0 R_t+\gamma^1R_{t+1}+\gamma^2R_{t+2}+...$ ,其中 $\gamma$ 是一个超参数。在这里，$U_t$ 也是个位置变量，因为动作还没有发生，我们没有办法获得 $t$ 时候的奖励以及 $t$ 时刻之后的奖励，所以 $R$ 都是随机的，那么我们的 $U_t$ 也是随机的，因为下面的第七点`强化学习的随机性`，所以我们在这里使用$R$来表示奖励，因为它是随机变量。
   4. 根据上面的分析，我们可以得知，$R_i$由$S_i$和$A_i$决定，那么$U_t$由一系列随机变量决定：$A_t,A_{t+1},A_{t+2},\dots \ and \ S_t,S_{t+1},S_{t+2},\dots $

7. 强化学习的随机性

   1. `动作具有随机性`，$\pi(\theta)$只输出各个动作的概率，动作是根据概率随机抽样而选取的，即$\mathbb{P}[A=a \vert S=s] =\pi(a\vert s)$
   2. `状态转换具有随机性`，并不是说在状态 $s_i$ 的情况下选取动作 $a$ 就一定会转移到一个固定的状态 $s_{i+1}$，这个状态也是随机的，他可能是 $s_1,s_2,s_3.....$中的任意一个，即$\mathbb{P}[S^{\prime}=s^{\prime}\vert S=s,A=a]=p(s^{\prime}\vert s,a)$

8. 轨迹 trajectory

   我们把一轮游戏从开始到结束的动作、状态、奖励拼起来也就是$(s_1,a_1,r_1,s_2,a_2,r_2,s_3,a_3,r_3, \dots, s_n,a_n,r_n)$这就是个轨迹，称之为 `trajectory,轨迹`，后面很多算法要用到`轨迹`

9. AI 如何控制 agent 打游戏？

   1. `学习 Q*`$(s_t,a_t)=max_\pi Q_{\pi}(s_t,a_t)$，根据状态 $s_t$ 选择 $a_t$，$a_t$ 满足条件：$a_t$ 能够使得 Q\*最大
   2. `学习策略 Pi(a|s)`，根据状态 $s_t$，根据 $\pi(·|s_t)$的概率随机采样
   3. 第一个就是ValueBased RL,第二种就是PolicyBased RL。

10. 概率论相关的数学知识

    1. 随机变量

       一个变量，它的值由一个随机事件决定，用大 X 表示随机变量，使用小 x 表示这个随机变量的观测值，`概率统计中统一使用大小写来标记随机变量以及他的观测值`

    2. 概率密度函数

       Probability Density Function，表示随机变量在某个附近的取值点的`可能性`。像高斯分布（正态分布）的函数就是一个概率密度函数。

    3. 期望

       给定 X 为随机变量，求 $f(X)$的期望：

       - 在离散情况下，就是 $p(x)f(x)$的加和
       - 在连续情况下，就是 $P(x)f(x)$的积分即$\int_x P(x)f(x)$

    4. 随机抽样

       获得 $X$ 的观测值 $x$ 的操作叫做随机抽样

    5. 蒙特卡洛 Monte Carlo 抽样的用法

       - 计算 $\pi$

         假定$(x,y)$是在一个边长为 $1$ 的正方形之内随机选一个点，那么这个点符合均匀分布的规律，那么这个点落在正方形内接圆的概率是多少呢？用面积可以算出来是 $π/4$,那我们抽样 $n$ 个点，应该有 $πn/4$ 个点落在圆里面，如果 $n$ 非常大的话我们发现 $m$ 个点在圆里面，那么 $m≈πn/4$。

         `要保证抽样是均匀的`

       - Buffon's Needle Problem

         投针问题也能够很好地近似估算 $\pi$

       - 估计阴影部分的面积

         使用蒙特卡洛进行近似计算

       - 近似求积分

         有些函数过于复杂，没有解析的积分，需要蒙特卡洛方法求定积分，也是无限次抽样

       - **近似期望**

         X 是一个 d 维的随机变量，p(x)是概率密度函数，平均分布的概率是 $p(x)=1/t  \ for \ x\in[0,t]$

         高斯分布/正态分布：$p(x)=1/(\sigma (2π)^2)\exp[-(x-\mu)^2/2\sigma^2]$

         直接求 $F(x)$关于 $P(x)$的定积分有时候很难，我们抽按照 $p(x)$的分布抽 $n$ 个样本，计算 $Q_n=\sum \frac {f(x_i)} {n}$，即 $Q_n$ 是 $ \mathbb{E}[f(x)]$
11. 关于TD学习

    TD 学习 temporal difference,与蒙特卡洛方法类似，时差(TD)学习是一个无模型(model free)方法，它从每轮的经验数据中学习。不同的是，TD 学习可以从不完整的一轮数据中学习，因此我们无需让代理一直执行到环境为终止态。

---

## 2. 价值学习 Value Based Leaning --学习 $Q^*(s,a)$

- 关于Qlearning的收敛性证明，请查看[这里](./QLearningReconstruction/readme.md)

- $U_t$ 被定义为折扣回报或者是折扣奖励，那么我们关于策略 π 的动作-价值函数 $Q_{\pi}(s_t,a_t)$等于 $U_t$ 的期望（因为 $U_t$ 求不出来，所以要求期望），叫做期望回报。

- 那么当前的 $Q_{\pi}$ 只与当前的状态和动作 $s_t$ 和 $a_t$ 有关，它反映了当前这个状态下执行动作 $a_t$ 的好坏

- $Q^*(s,a)$ 为当策略最好的时候我们的动作状态值(optimal action-value function)，也就是说，不管我们使用什么策略 $\pi$，我们最后选取的动作，他的 Q 值都不会比 Q*好, $Q^*$ 函数的意义是指示Agent在 $s$ 状态下选取动作 $a$ 时的好坏。

- 难点在于我们不知道所谓的 $Q^*(s,a)$ ，于是我们使用深度神经网络 $Q(s,a;\mathbf{w})$ 去拟合 $Q^*(s,a)$ ,对于不同的问题，神经网络的结构也可能不同。

- 关于 TD 学习 temporal difference Learning：
  - $Q(\mathbf{w})$负责估计代价，我们采样 $q=Q(\mathbf{w})$，假设采样值为1000
  - 在现实中，我们进行试验，比如说玩一整轮游戏，然后得到实际的代价 $y=860$
  - 计算损失$\mathcal{L}=\frac{(q-y)^2}{2}$
  - 计算$\mathcal{L}$ 关于参数$\mathbf{w}$ 的梯度，根据链式求导法则，我们可以得到 $
  \frac{\partial{\mathcal{L}}}{\partial \mathbf{w}}=\frac{\partial q}{\partial \mathbf{w}}\frac{\partial{\mathcal{L}}}{\partial q}=(q-y)\frac{\partial Q(\mathbf{w})}{\partial \mathbf{w}}$
  - 进行梯度下降，$w_{t+1} = w_t - \alpha \frac{\partial{\mathcal{L}}}{\partial \mathbf{w}}|_{\mathbf{w}=\mathbf{w}_t}$，其中 alpha 是超参数，是步长。
  - ![td learning](./DeepQLearningNetwork(DQN)/TDLearningNaive.png)
  - 但是在玩游戏的过程中，我们因为某种原因，只玩到一半，得到价值，我们需要$ Q(w)$估计另外一半的代价，两者相加得到代价 $\hat y$，这个 $\hat y$称为`TD target`，它肯定比 $Q(w)$估计整个过程要靠谱，因为我们有一半的数值是真的。我们用这个 $\hat y$ 来代替上面的 $y$，也可以更新参数。
  - 由上一步，我们将$Q(\mathbf{w})-\hat y$称为`TD ERROR, Temporal Difference Error`
  - 我们的优化目标就是让 TD Error = 0

### 1. Qlearning - off_policy TD control

建议**对比下面的[Sarsa](# 2. Sarsa (State-Action-Reward-State-Action) - on_policy TD control)算法来看**

- QLearning 训练`最优的动作-价值函数` $ Q^*(s,a)$，TD Target是 $y_t=r_t + \gamma \ \underset{a}{max} Q^*(s_{t+1},a) $，DQN就是这个模式。

- 维护一个 Q 表，表中的每个元素代表每个状态下每个动作的潜在奖励
根据 Q 表选择动作，然后更新 Q 表

  ```text
  state 1 2 3 4 5
  left  0 0 0 0 0
  right 0 0 0 1 0
  ```

- 更新策略：`现实值=现实值+lr*（估计值-现实值）`

#### 1. **推导**

- 对于所有的策略 $\pi$ 有
  $$
  Q_{\pi}(s_t,a_t) = \mathbb{E}[R_t+\gamma \cdot Q_{\pi}(S_{t+1},A_{t+1})]
  $$
  对于最优的策略 $\pi^*$来说也有
  $$
  Q_{\pi^*}(s_t,a_t) = \mathbb{E}[R_t+\gamma \cdot Q_{\pi^*}(S_{t+1},A_{t+1})]
  $$

- 在QLearning中，我们使用
  $$ A_{t+1} = \underset{a}{argmax} \ Q^*(S_{t+1},a)$$
  来计算 $ A_{t+1} $，所以我们有
  $$
  Q_{\pi^*}(S_{t+1},A_{t+1}) = \underset{a}{max} \ Q^*(S_{t+1},a)
  $$

- 消掉 $Q_{\pi^*}(S_{t+1},A_{t+1})$ ,于是我们消掉了 $A_{t+1}$，于是我们有
  $$
  Q_{\pi^*}(s_t,a_t) = \mathbb{E}[R_t+\gamma \cdot \underset{a}{max} \ Q^*(S_{t+1},a)]
  $$

- 期望很难求，于是又要做蒙特卡洛近似，使用观测值$r_t$,$s_{t+1}$来近似$R_t$,$S_{t+1}$，于是有：
  $$
  Q_{\pi^*}(s_t,a_t) = \mathbb{E}[r_t+\gamma \cdot \underset{a}{max} \ Q^*(s_{t+1},a)]
  $$
  我们称为 $r_t+\gamma \cdot \underset{a}{max} \ Q^*(s_{t+1},a)$叫做TD Target $y_t$

#### 2. **算法步骤(表格形式)**

1. 观测到状态 $(s_t,a_t,r_t,s_{t+1})$
2. 计算TD Target : $r_t+\gamma \cdot \underset{a}{max} \ Q^*(s_{t+1},a)$
3. 计算 TD Error : $ \delta_t = Q^*(s_t,a_t)-y_t$
4. 更新$Q^*$ : $Q^*(s_t,a_t) \gets Q^*(s_t,a_t)  - \alpha \cdot \delta_t$
5. 根据 $\underset{a}{max} \ Q^*(s_{t+1},a)$采样动作，然后采取该动作，转1

### 2. Sarsa (State-Action-Reward-State-Action) - on_policy TD control

#### 1. **与QLearning的区别**

- Qlearning 更新方法：`根据当前Q表选择动作->执行动作->更新Q表`

- Sarsa 更新方法：`执行动作->根据当前估计值选择下一步动作->更新Q表`

- 总结：**Sarsa 是行动派，Qlearning 是保守派**
- Sarsa 训练`动作-价值函数` $Q_{\pi}(s,a) $ ，它的 TD Target是 $y_t= r_t + \gamma \cdot Q_{\pi}(s_{t+1},a_{t+1})$, Sarsa是更新价值网络(Critic)

- QLearning 训练`最优的动作-价值函数` $ Q^*(s,a)$，TD Target是 $y_t=r_t + \gamma \ \underset{a}{max} Q^*(s_{t+1},a) $，DQN就是这个模式。

如果状态空间很大的话我们的表格就很大，如果是连续的动作或者连续的状态的话，就不能用表格来表示了，这时可以使用神经网络来近似状态-价值函数 $Q_{\pi}(s,a)$

- 由前面的推导我们可以知道
  $$
  U_t = R_t + \gamma \cdot U_{t+1}
  $$
- 假设 $R_t$ 由 $(S_t,A_t,S_{t+1})$决定
- 那么可以推导出
  $$
  \begin{aligned}
  Q_{\pi}(s_t,a_t) &= \mathbb{E}[U_t \vert s_t,a_t]\\
  &=\mathbb{E}[R_t + \gamma \cdot U_{t+1} \vert s_t,a_t]\\
  &=\mathbb{E}[R_t\vert s_t,a_t] + \gamma \cdot \mathbb{E}[U_{t+1} \vert s_t,a_t]\\
  &=\mathbb{E}[R_t\vert s_t,a_t]+ \gamma \cdot \mathbb{E}[Q_{\pi}(S_{t+1},A_{t+1}) \vert s_t,a_t]
  \end{aligned}
  $$
- 期望很难算，所以又要做蒙特卡洛近似，使用 $r_t$和 $Q_{\pi}(s_{t+1},a_{t+1})$去近似 $R_t$ 和 $Q_{\pi}(S_{t+1},A_{t+1})$
- 于是就有了
  $$ Q_{\pi}(s_t,a_t) \approx r_t + \gamma \cdot Q_{\pi}(s_{t+1},a_{t+1})$$
  我们把 $r_t + \gamma \cdot Q_{\pi}(s_{t+1},a_{t+1})$ 称为**TD Traget** $y_t$
- TD Learning 的想法就是鼓励 $Q_{\pi}(s_t,a_t)$向 $y_t$逼近

#### 2. **算法步骤(表格形式的Sarsa)**

1. 观测到状态 $(s_t,a_t,r_t,s_{t+1})$
2. 采样动作 $ a_{t+1} \sim \pi(\cdot \vert s_{t+1})$ 其中 $\pi$是策略函数
3. 计算TD Target $y_t = r_t + \gamma \cdot Q_{\pi}(s_{t+1},a_{t+1})$
4. 计算TD Errorr $\delta_t = Q_{\pi}(s_{t},a_{t}) - y_t $
5. 更新 $Q_{\pi}(s_{t},a_{t})$:  $Q_{\pi}(s_{t},a_{t}) \gets Q_{\pi}(s_{t},a_{t}) - \alpha\cdot\delta_t$ 其中$\alpha$是学习率，在神经网络中，我们采用**梯度下降的方式**来更新$Q_{\pi}(s_{t},a_{t})$
6. 执行$a_{t+1}$转步骤1

#### 3. **使用多步TD Target 来减少偏差**

  之前说到$ U_t = R_t + \gamma U_{t+1}$,我们可以进一步展开：
  $$
  U_t = R_t + \gamma (R_{t+1}+\gamma U_{t+2})
  $$
  这样就可以使用2步甚至更多步的数据来更新我们的神经网络，来提升稳定性。

### 3. SarsaLambda

是*Sarsa 的升级版*

- Qlearning 和 Sarsa 都认为上一步对于成功是有关系的，但是上上一步就没有关系了，SarsaLambda 的思想是：`到达成功的每一步都是有关系的，他们的关系程度为：越靠近成功的步骤是越重要的`

  ```text
  step
  1-2-3-4-5-success
  重要性1<2<3<4<5
  ```

### 4. DQN Off-Policy

- 神经网络 $Q(s,a;\mathbf{w})$近似 Q\*函数，Q\*能够告诉我们每个动作能够得到的平均回报。我们需要 agent 遵循这个 Q\*函数。

- 用神经网络代替 Q 表的功能![dqb algo](<./DeepQLearningNetwork(DQN)/dqn.jpg>)

- Q 表无法进行所有情况的枚举，在某些情况下是不可行的，比如下围棋。

#### 1.  Features: `Expericence Replay and Fixed Q-targets`

- Experience Replay : `将每一次实验得到的经验片段记录下来，然后作为经验，投入到经验池中，每次训练的时候随机取出一个 BATCH，可以复用数据。并且可以在经验池上面做一些文章，增加收敛性比如HER、PER、ERE等等。`

- Fixed Q-target: `在神经网络中，Q 的值并不是互相独立的，所以不能够进行分别更新操作，那么我们需要将网络参数复制一份，解决该问题。`

- 为了解决 `overestimate` 的问题，引入 `double DQN`，算法上有一点点的改进，复制一份网络参数，两个网络的参数`异步`更新
  - 在强化学习中，我们使用于Bootstrapping来更新网络参数，因为TD target和$Q(s_t,a_t;\mathbf{w})$都有估计的成分，我们更新网络参数$\mathbf{w}$的时候是用一个估计值来更新它本身，类似于自己把自己举起来。
  - TDLearning会使DQN高估动作价值(overestimate),原因在于：
    - 1.Qlearning中TD target中有最大化操作：
    $ y_t = r_t + \gamma \cdot \underset{a}{max} \ Q(s_{t+1},a;\mathbf{w})$这个操作会导致overestimating

      设$x(a_1),\dots,x(a_n)$为真实的动作价值，$a_1,\dots,a_n$是$a \in \mathcal{A}$中的所有动作。

      我们使用DQN来对上面的动作价值做有噪声的估计：$Q(s,a_1;\mathbf{w}),\dots,Q(s,a_n;\mathbf{w})$

      假设这个估计是无偏差的(unbiased estimation)：$\underset{a}{mean}(x(a)) = \underset{a}{mean}(Q(s,a;\mathbf{w}))$

      而$q = \underset{a}{max}Q(s,a,\mathbf{w})$,所以我们有 $q \ge \underset{a}{max}(x(a))$(因为Q的估计有偏差，所以它的最大值应该是大于或等于真实值的最大值)

      ***总结 ： 求最大化使估计值大于实际值，导致overestimate的问题。***
    - 2.Bootstrapping使用估计值更新自己的话，会传播这个高估的值。

      我们首先回顾TD target需要用到下一时刻的估计：$ q_{t+1} = \underset{a}{max}Q(s_{t+1},a;\mathbf{w}) $,然后我们使用TD target 来更新我们的$ Q(s_t,a_t;\mathbf{w}) $

      假设DQN已经因为最大化高估了动作价值(action-value)，由于$Q(s_{t+1},a;\mathbf{w})$已经高估了，然后还要使用$ q_{t+1} = \underset{a}{max} \ Q(s_{t+1},a;\mathbf{w}) $中的max操作，这就导致了更严重的高估，然后使用$r_t + \gamma q_{t+1}$来更新，传播了这个高估，变得更严重了。

      ***总结：TD target本身有max操作，产生更严重的高估，用TD target更新DQN又进一步加剧了高估***
  - 为什么Overestimating会带来不好的影响？

    如果DQN将每个动作的价值都高估了一个一样的值的话，我们通过max操作得出一样的结果：

    假设$Q(s,a^i;\mathbf{w})= Q^*(s,a^i) + 100$，我们使用max操作还是能够得到一样的结果

    但是如果高估是`非均匀`的话，那么就会影响最后的max操作的结果了，这样我们就会基于`错误的价值`进行决策。

    很不幸，在ReplayBuffer里面，我们这种高估是`非均匀`的：状态$(s_t,a_t)$这个二元组每一次被抽样到就会让DQN高估$(s_t,a_t)$的值，越被频繁抽样到就会产生越严重的高估。而$s$和$a$在经验池中的频率是不均匀的，最终会导致不均匀的高估，他是非常有害的。
  - 如何缓解高估问题？
    1. 使用`Fixed Q-targets`避免bootstrapping。

        - 我们使用两个神经网络$Q(s,a;\mathbf{w}^-)，Q(s,a;\mathbf{w})$来近似动作价值函数，这两个神经网络有一样的结构，但是它们的参数是不同的。

        - 使用$Q(s,a;\mathbf{w})$来控制agent来收集经验$\{(s_t,a_t,r_t,s_{t+1}\}$

        - 使用target networtk $Q(s,a;\mathbf{w}^-)$来计算TD Target，从而避免了bootstrapping，缓解了高估问题。

        - 具体的来说，我们计算TD target用的是Target Network: $y_t = r_t + \gamma \cdot \underset{a}{max} Q(s_{t+1},a;\mathbf{w}^-)$,然后计算TD Error $\delta_t = Q(s_t,a_t;\mathbf{w}) - y_t$,再执行梯度下降更新原来网络的参数$\mathbf{w}$的权重，不更新$\mathbf{w}^-$这样就避免了自举。

        - 更新Target Network的方式有两种：一种是直接复制$\mathbf{w}$到$\mathbf{w}^-$上，另外一种是soft_update，将两个参数进行加权平均：$\mathbf{w}^- \gets \tau \mathbf{w}^- + (1-\tau)\mathbf{w} $,其中$\tau$一般取比较保守的值0.95,0.9这些。

        - 总结：使用Fixed Q-targets来计算TD Target 避免bootstrapping，但是Target Network无法脱离原来的网络，上面看到了，Target Network的更新是与原来的Network有关的，所以它无法完全避免自举。

    2. 使用`Double DQN`来缓解最大化造成的高估。

        - 与Fixed Q-targets的区别是计算TD Target的时候使用的网络不一样，前者使用：
          $$
          \begin{aligned}
          a^*&= \underset{a}{argmax} Q(s_{t+1},a;\mathbf{w}^-)\\
          y_t &= r_t +\gamma \cdot Q(s_{t+1},a^*;\mathbf{w}^-)
          \end{aligned}
          $$
          来计算TD Target，后者使用
          $$
          \begin{aligned}
          a^*&= \underset{a}{argmax} Q(s_{t+1},a;\mathbf{w})\\
          y_t &= r_t +\gamma \cdot Q(s_{t+1},a^*;\mathbf{w}^-)
          \end{aligned}
          $$
          double dqn做出的改动很小，但是性能提升很大，然而它还是没有彻底解决高估的问题。
        - 为什么它比前面的Fixed Q-targets要好？

          因为$ Q(s_{t+1},a^*;\mathbf{w}^-) \le \underset{a}{max} \ Q(s_{t+1},a;\mathbf{w}^-)$

#### 2. TD 算法在 DQN 中的使用

- 类似于我们在本章开头中提出 TD 学习的概念，我们在 DQN 中也有：$Q(s_t,a_t;\mathbf {w})≈r_t + \gamma Q(s_{t+1},a_{t+1};\mathbf {w})$
- 在上式中，$\gamma$ 为一个奖励的折扣因子
- 折扣回报：$U_t= R_t + \gamma ((R_{t+1})+\gamma(R_{t+2})+...)$ --(在前面消掉一个$\gamma$)
- 那么我们的折扣回报可以写成 $U_t = R_t + \gamma U_{t+1}$, 因为$ \gamma U_{t+1}=\gamma ((R_{t+1})+\gamma(R_{t+2})+...)$
- 反映了两个相邻状态之间的折扣回报的关系
- 那么我们使用 DQN 来输出这个 $U_t$ 的期望（说过很多次，在 $t$ 时刻之后，动作 $A$ 和状态 $S$ 都是随机变量，所以求期望）
- 我们有了$U_t = R_t + \gamma U_{t+1}$，而且$Q(s_t,a_t;\mathbf{w})$是$\mathbb{E}[U_t]$的估计， $Q(s_{t+1},a_{t+1};\mathbf{w})$是$\mathbb{E}[U_{t+1}]$的估计,所以我们有
    $$
    Q(s_t,a_t;\mathbf{w}) ≈ \mathbb{E}[R_t + \gamma Q(s_{t+1},a_{t+1};\mathbf{w})]
    $$
- 到了$t$时刻，我们已经获得观测值 $r_t $了，所以有$Q(s_t,a_t;\mathbf{w}) ≈ r_t +\gamma Q(s_{t+1},a_{t+1};\mathbf{w})$,约等于号后面的那个值肯定要准确一些，我们称之为 TD target , 前面$Q(s_t,a_t;\mathbf{w})$是 prediction（预测值）
- 于是我们的 $loss = \frac{1}{2} ||predict - target ||_2$，再进行梯度下降就可以了

- 使用Experience Replay的动机
  
  - 一个Transition $(s_t,a_t,r_t,s_{t+1})$使用完之后就把它丢弃掉了，不再使用，这是一种浪费，经验可以被重复使用。
  - $s_t$和$s_{t+1}$是高度相关的，这样不利于我们模型的收敛，所以需要把$s_t,s_{t+1},\dots,s_T$这个序列打散。
  - 于是我们将最近$n$个transitions放到一个`经验池中`,$n$是一个超参数，他一般是十万或者百万级别的。
  - 在ER里面，可以做mini-batch SGD，随机均匀抽取一小部分样本，取梯度的平均值进行梯度下降。
  - 除了均匀抽样以外，我们还有非均匀抽样，这也就是下面的[PrioritizedExperienceReplay](#6-dqn-with-prioritized-experience-replay-off-policy)

#### 3.三种方法的价值网络的更新方式比较

|         method         | selection |evaluation|
| :--------------------: | ---- |--|
|Naive DQN|$\mathbf{w}$|$\mathbf{w}$|
|Fixed Q-targets|target network $\mathbf{w}^-$|target network $\mathbf{w}^-$|
|double DQN|$\mathbf{w}$|target network $\mathbf{w}^-$|

### 5. Dueling DQN Off-Policy

将 Q 值的计算分成状态值 state_value 和每个动作的值 advantage，可以获得更好的性能，这是网络架构上面的改进，这个思想也可以用在其它地方。

#### 1. Advantage Function 优势函数

  $$
  \begin{aligned}
  Q^* (s,a) &= \underset{a}{max} \  Q_{\pi}(s,a)\\
  V^* (s) &= \underset{\pi}{max} \  V_{\pi}(s)\\
  A^* (s,a) &= Q^*(s,a) -V^*(s)
  \end{aligned}
  $$
  $A^*(s,a)$的意思是动作$a$相对于baseline $V^*(s)$的优势，动作$a$越好，$A^*(s,a)$越大。

  由于$ V^*(s) = \underset{a}{max} \ Q ^*(s,a)$，我们对左右两边取最大值，有：
  $$
  \underset{a}{max} \ A^*(s,a) = \underset{a}{max} \ Q_{\pi}(s,a) - V^*(s) =0
  $$

  我们将公式变换一下：
  $$
  Q^*(s,a) = V^*(s) + A^* (s,a)
  $$

  再减去一个0 ： $\underset{a}{max} \ A^*(s,a)$得到：
  $$
  Q^*(s,a) = V^*(s) + A^* (s,a) - \underset{a}{max} \ A^*(s,a)
  $$

#### 2. Dueling DQN的设计

- 我们使用神经网络$A(s,a;\mathbf{w}^A)$去近似$A^* (s,a)$

- 再使用$V(s;\mathbf{w}^V)$来近似$V^*(s)$

- 然后，我们的$Q^*(s,a)便可以用两个神经网络来表示：
  $$
  Q(s,a;\mathbf{w}^A,\mathbf{w}^V) = V(s;\mathbf{w}^V)+ A(s,a;\mathbf{w}^A) - \underset{a}{max} \ A(s,a;\mathbf{w}^A)
  $$

  它相比Naive DQN多了一个参数，训练过程是一样的，因为两者都需要采取状态$s$作为输入，我们一般共享feature层。

- 为什么上面的公式需要减上$\ \underset{a}{max} \ A(s,a;\mathbf{w}^A)$
  
  因为$ Q^*(s,a) = V^*(s) + A^*(s,a)$中满足这个条件的$V^*(s) + A^* (s,a)$有很多组，比如$ 1+9=10, 2+8=10$这样会使训练不稳定

  所以加上后面的最大化，防止这种不唯一性，防止上下波动导致的训练不稳定。
  $$
  Q^*(s,a) = V^*(s) + A^*(s,a) - \underset{a}{max} \ A^*(s,a)
  $$

  在实践中，我们会把max换成mean:
  $$
  Q^*(s,a) = V^*(s) + A^*(s,a) - \underset{a}{mean} \ A^*(s,a)
  $$

在[DuelingDQN的模型代码中](DuelingDQN\models.py)我们可以看到：

```python
def forward(self, x: t.Tensor) -> t.Tensor:
        feature1 = F.relu(self.feature_layer(x))
        feature2 = F.relu(self.feature_layer(x))
        # value
        value = self.value_layer(feature1)
        # advantage
        advantage = self.advantage_layer(feature2)
        # 其实在这里写advantage.mean(dim=1).expand_as(advantage)也是可以的
        # advantage满足 sum(advantage)=0
        return value+advantage-advantage.mean(dim=1, keepdim=True)
```

### 6. DQN with Prioritized Experience Replay Off-Policy

在 DQN 中，我们有 Experience Replay，但是这是经验是随机抽取的，我们需要让好的、成功的记忆多多被学习到，所以我们在抽取经验的时候，就需要把这些记忆优先给网络学习，于是就有了`Prioritized` Experience Replay。

`PER只在经验池这个地方做了改进，具体部分可以查看代码。`

- 使用TD Error来判断Transitions的重要性，如果DQN不熟悉这个transition，我们就让这个transition多多的出现，就能够让DQN熟悉。

- 使用重要性采样(importance sampling)来替代均匀抽样(uniform sampling):
  $ p_t \propto \vert \delta_t \vert +\epsilon$或者是 $ p_t \propto \frac{1}{rank(t)}$，其中$rank(t)$是td error第t大的transition。
- 两种方法的原理是一样的，就是让$\delta_t$越大的transition更多地被采样到。

- 不均匀抽样会导致偏差，我们需要对学习率进行缩放

  - 将lr缩放至 $ \gamma \gets \gamma \cdot (n p_t)^{-\beta}$ ，其中 $\beta \in (0,1)$
  - 拥有较高优先级（较高的 $p_t$ ）的transitions有较低的学习率；开始的时候 $\beta$ 很小，之后随着训练的进程提升到1。

- 如果新经验没有被学习，我们将它的$\delta_t$设置为$ \delta_{max}$，之后随着训练的进程，当这个经验被学习到的时候，我们更新这个$\delta_t$

## 3. 策略学习 Policy Based Learning --学习策略 $\pi(a|s)$

- 在连续空间下，State是无限的，使用基于Qlearning的方法需要求 $ max_{a \in \mathcal{A}} Q_{\pi}(s,a)$ 的计算代价很大，我们期望基于策略的方法在连续空间中的表现会比策略迭代法有用。

- Policy Function $\pi(a \vert s)$是一个概率密度函数(probability density function)，它以状态$s$为输入，输出的是每个动作对应的`概率值`。
在DDPG等算法中，我们的Policy是确定的，也就是它输出一个动作，而不是概率值，我们称之为 $ a = \mu(s)$

- Discounted Return, Action-value function, State-value function
  $$
  \begin{aligned}
  U_t &= R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \gamma^3 R_{t+3} + \dots
   \\

  Q_{\pi}(s_t,a_t) &= \mathbb{E}[U_t \vert S_t=s_t,A_t= a_t]\\

  V_{\pi}(s_t)&=\mathbb{E}_
  A[Q_{\pi}(s_t,A)], A \sim \pi (\cdot \vert s_t)

  \end{aligned}
  $$
  对于离散的动作我们有
  $$
  V_{\pi}(s_t) = \mathbb{E}[Q_{\pi}(s_t,A)] =\Sigma_a \pi(a \vert s_t)Q_{\pi}(s_t,a),A \sim \pi (\cdot \vert s_t)
  $$
  对于连续的动作，我们需要求个积分
  $$
  V_{\pi}(s_t) = \mathbb{E}[Q_{\pi}(s_t,A)] =\int \pi(a \vert s_t)Q_{\pi}(s_t,a),A \sim \pi (\cdot \vert s_t)
  $$

- 在基于策略的方法里，我们需要使用神经网络$\pi (a\vert s_t;\mathbf{\theta})$来近似`策略函数`$\pi(a\vert s_t)$,使用$V(s_t;\mathbf{\theta})=\Sigma_a \pi(a \vert s_t;\mathbf{\theta})Q_{\pi}(s_t,a)$来`状态价值函数state-value function`

- 在Policy-based methods里面，我们要尝试最大化$V(s;\mathbf{\theta})$的期望。
  即$J(\mathbf{\theta})=\mathbb{E}_S[V(s;\mathbf{\theta})]$

- 使用`梯度上升`方法来更新$\theta$

  观测到状态$s$

  Update Policy by: $\theta \gets \theta + \beta \cdot \frac{\partial V(s;\theta)}{\partial \theta}$

  其中这个$\frac{\partial V(s;\theta)}{\partial \theta}$就叫做`策略的梯度Policy Gradient`,详细的推导请见下面的[Policy Gradient方法](#1-policy-gradient-on-policy)

- 在这个章节中，除了 Policy Gradient 算法没有用到 Critic，其余好像都用到了 critic 或者类似于 actor-critic 的架构，比如说 DDPG 是个 AC 架构，而 AC A2C A3C TRPO 等都用到了 Actor 和 Critic，两者的区别就是Actor参数更新方式不同，AC架构仍然使用Q Value作为Actor的loss，而AC就是使用带权重的梯度更新。

- PG 算法学习的就是策略，像[PG 中 readme](<https://github.com/Phoenix-Shen/ReinforcementLearning/tree/main/PolicyGradient(PG)#%E6%9B%B4%E6%96%B0%E7%BD%91%E7%BB%9C%E5%8F%82%E6%95%B0>)里面说的一样我们为什么不用神经网络来近似 $Q_{\pi}$，就可以不用 Discounted Rewards 来代替 $Q_{\pi}$。

- 所以我们可以转为策略学习+值学习，他是 Value based methods 和 Policy based methods 的结合

- 状态价值$ V_{\pi}(s)=\Sigma_a \pi(a|s) Q_{\pi}(s,a)$，使用 Actor 来近似 $\pi$，使用 Critic 来近似 $Q_{\pi}$.

- 减少方差--策略梯度方法中的`baseline`
  
  1. baseline $b$可以是任何独立于动作$A$的函数

  2. 证明baseline的理论性质
      $$
      \begin{aligned}
      \mathbb{E}
      _
      {A \sim \pi(\cdot \vert s; \theta)}\left[b \cdot \frac{\partial  \ln \pi(A \vert s;\theta) }{\partial \theta} \right] &= b \cdot \mathbb{E}_ {A \sim \pi(\cdot \vert s; \theta)}\left[\frac{\partial  \ln \pi(A \vert s;\theta) }{\partial \theta} \right]\\

      &= b \cdot \sum_a \pi(a \vert s; \theta) \cdot \frac{\partial  \ln \pi(a \vert s;\theta) }{\partial \theta}\\

      &= b \cdot \sum_a \pi(a \vert s; \theta) \cdot \frac {1}{\pi(a \vert s; \theta)}\frac{\partial   \pi(a \vert s;\theta) }{\partial \theta}\\

      &= b \cdot \sum_a \frac{\partial   \pi(a \vert s;\theta) }{\partial \theta}\\

      &= b \cdot  \frac{\sum_a \partial   \pi(a \vert s;\theta) }{\partial \theta}\\

      &= b\cdot \frac{\partial 1 }{\partial \theta}\\

      &= 0
      \end{aligned}
      $$

      于是有策略梯度公式：
      $$
      \begin{aligned}
      \frac{\partial V(s;\theta)}{\partial \theta}&=
        \mathbb{E}_
        {A \sim \pi(\cdot \vert s; \theta)}\left[\frac{\partial  \ln \pi(A \vert s;\theta) }{\partial \theta} Q_{\pi}(s,A)\right]\\
      &= \mathbb{E}_
        {A \sim \pi(\cdot \vert s; \theta)}\left[\frac{\partial  \ln \pi(A \vert s;\theta) }{\partial \theta} Q_{\pi}(s,A)\right] - \mathbb{E}_ {A \sim \pi(\cdot \vert s; \theta)}\left[\frac{\partial  \ln \pi(A \vert s;\theta) }{\partial \theta} \right]\\

      &=\mathbb{E}_
        {A \sim \pi(\cdot \vert s; \theta)}\left[\frac{\partial  \ln \pi(A \vert s;\theta) }{\partial \theta} \left(Q_{\pi}(s,A)-b\right)\right]
      \end{aligned}
      $$

      所以我们有：如果$b$与$A_t$是独立的，那么策略梯度可以表示为：
      $$
      \mathbb{E}_
        {A_t \sim \pi(\cdot \vert s_t; \theta)}\left[\frac{\partial  \ln \pi(A_t \vert s_t;\theta) }{\partial \theta} \left(Q_{\pi}(s_t,A_t)-b\right)\right]
      $$
      baseline $b$不会影响上面的结果，在我们对这个梯度进行蒙特卡洛近似的时候，一个好的$b$会让后面的$\left(Q_{\pi}(s_t,A_t)-b\right)$会让方差降低，算法收敛更快

  3. 蒙特卡洛近似

      现在我们知道了策略梯度：
      $$
      \frac{\partial V(s;\theta)}{\partial \theta}=\mathbb{E}_
        {A_t \sim \pi(\cdot \vert s_t; \theta)}\left[\frac{\partial  \ln \pi(A_t \vert s_t;\theta) }{\partial \theta} \left(Q_{\pi}(s_t,A_t)-b\right)\right]
      $$
      令
      $$
      \left[\frac{\partial  \ln \pi(A_t \vert s_t;\theta) }{\partial \theta} \left(Q_{\pi}(s_t,A_t)-b\right)\right] = \mathbf{g}(A_t)
      $$

      我们随机抽样动作$a_t$ : $a_t \sim \pi(\cdot \vert s_t; \theta)$然后计算梯度$\mathbf{g}(a_t)$，而且$\mathbf{g}(a_t)$是对原来梯度的一个无偏估计:
      $$
      \mathbb{E}_
        {A_t \sim \pi(\cdot \vert s_t; \theta)}[\mathbf{g}(a_t)] \approx \frac{\partial V_\pi(s_t;\theta)}{\partial \theta}
      $$

      然后执行梯度上升
      $$
      \theta \gets \theta + \beta\cdot \mathbf{g}(a_t)
      $$

      前面证明了只要$b$跟$A_t$无关，我们的$\mathbf{g}(a_t)$的期望$\mathbb{E}_{A_t \sim \pi(\cdot \vert s_t; \theta)}[\mathbf{g}(A_t)]$就不会变，但是$b$会影响抽样结果$\mathbf{g}(a_t)$，所以一个好的$b$会使方差减少，增加算法收敛

  4. baseline的选取

      - $b=0$

        这就是最基本的policy gradient方法

      - $b = V_{\pi}(s_t)$

        状态$s_t$先被观测到，与$A_t$无关，由于$V_{\pi}(s_t) = E_{A_t}[Q_\pi(s_t,A_t)]$，所以用$V_{\pi}(s_t)$是很合适的

- `策略梯度算法的一般形式`
  ![genaral advantage estimate](./PolicyGradient(PG)/gae.png)

### 1. Policy Gradient On-Policy

核心思想：让好的行为多被选择，坏的行为少被选择。

![PG](<./PolicyGradient(PG)/5-1-1.png>)

#### 1. 具体推导（简单版本）

  这里是不严谨的推导，便于直观地上手，我们假设 $Q_\pi$ 不依赖于 $\theta$，要查看详细的推导，请移步至
  [本文的这里](#4-policy-gradient-算法的详细推导有点难),具体的过程以及一系列讲解请查看
  [这个网站](https://paperexplained.cn/articles/article/detail/31/)

  $$V(s_t;\mathbf{\theta})=\Sigma_a \pi(a \vert s_t;\mathbf{\theta})Q_{\pi}(s_t,a)$$

  $$
  \begin{aligned}
  \frac{\partial V(s;\theta)}{\partial \theta} &= \frac{\partial \Sigma_a \pi(a \vert s;\theta) Q_{\pi}(s,a)}{\partial \theta}\\

  &= \Sigma_a\frac{\partial  \pi(a \vert s;\theta) Q_{\pi}(s,a)}{\partial \theta}\\

  &= \Sigma_a\frac{\partial  \pi(a \vert s;\theta) }{\partial \theta} Q_{\pi}(s,a) \text{ 假设Qpi不依赖于theta,但不严谨}\\

  \end{aligned}
  $$
  于是就有了
  $$
  \begin{aligned}

  \frac{\partial V(s;\theta)}{\partial \theta}&=\Sigma_a\frac{\partial  \pi(a \vert s;\theta) }{\partial \theta} Q_{\pi}(s,a)\\

  &= \Sigma_a\ \pi(a \vert s;\theta)\frac{\partial  \log \pi(a \vert s;\theta) }{\partial \theta} Q_{\pi}(s,a)\\
  
  &= \mathbb{E}_ {A \sim \pi(\cdot \vert s; \theta)}\left[\frac{\partial  \log \pi(A \vert s;\theta) }{\partial \theta} \ Q_{\pi}(s,A)\right]
  \end{aligned}
  $$

- 对于离散的动作来说使用
        $$
        \frac{\partial V(s;\theta)}{\partial \theta}=\Sigma_a\frac{\partial  \pi(a \vert s;\theta) }{\partial \theta} Q_{\pi}(s,a)
        $$
        对于每个动作都求一次，然后加起来就可以辣

- 对于连续的动作来说使用
        $$
        \frac{\partial V(s;\theta)}{\partial \theta}=
        \mathbb{E}_{A \sim \pi(\cdot \vert s; \theta)}\left[\frac{\partial  \log \pi(A \vert s;\theta) }{\partial \theta} Q_{\pi}(s,A)\right]
        $$

- 使用蒙特卡洛抽样来求梯度

  1. 根据概率密度函数$\pi (\cdot \vert s;\theta)$采样出一个$\hat a$
  2. 计算$\mathbf{g}(\hat a ,\theta) = \frac{\partial \log \pi(\hat a \vert s;\theta) }{\partial \theta} Q_{\pi}(s,\hat a)$
  3. 很显然，$\mathbb{E}_A [\mathbf{g}(A,\theta)]=\frac{\partial V(s;\theta)}{\partial \theta}$而且$\mathbf{g}(\hat a ,\theta)$是$\frac{\partial V(s;\theta)}{\partial \theta}$的一个无偏估计。

  这种方法对于上面的离散行为也适用

#### 2. Policy Gradient**算法细节**

  1. 观测到当前状态$s_t$

  2. 根据策略网络$\pi (\cdot \vert s; \theta_t)$来选取一个动作$a_t$,注意动作$a_t$是随机抽样得来的

  3. 计算$q_t \approx Q_{\pi}(s_t,a_t)$，在这一步需要做一些估计

  4. 求$J(\theta)$关于$\theta$的梯度$\mathbf{g}(a_t ,\theta_t) = q_t \frac{ \log \pi(a_t \vert s_t;\theta) }{\partial \theta} |_{\theta =\theta_t}$

  5. 梯度上升:$\theta_{t+1} = \theta_t + \beta \cdot \mathbf{g}(a_t,\theta_t)$

  在第三步中，我们如何计算 $q_t \approx Q_{\pi}(s_t,a_t)$ ? 有两种方法：

  1. REINFORCE

      玩一局游戏得到这局游戏的轨迹Trajectory

      $s_1,a_1,r_1,s_2,a_2,r_2,\dots,s_T,a_T,r_T$

      对于所有的$t$计算discounted return
      $
      u_t = \sum_{k=t}^T \gamma^{k-t}r_k
      $

      由于$Q_{\pi}(s_t,a_t) = \mathbb{E}[U_t]$,我们可以使用$u_t$来去近似$Q_{\pi}(s_t,a_t)$，这种方法显而易见有一个缺点：玩完一局游戏才能进行更新，低效。
  2. 使用神经网络去近似$Q_{\pi}$

      这就是下面的[ActorCritic Methods](#2-actor-critic-on-policy-and-advantage-actor-critic-on-policy)

#### 3. REINFORCE with Baseline

- 关于baseline可以在[策略学习](#3-策略学习-policy-based-learning---学习策略-pias)这里看到

- 我们有随机策略梯度：
    $$
    \mathbf{g}(a_t) = \frac{\partial \ln \pi(a_t \vert s_t;\theta) }{\partial \theta} \cdot \left(Q_\pi (s_t,a_t)-V_\pi(s_t) \right)
    $$

- 由于 $Q_\pi(s_t,a_t) = \mathbb{E}[U_t \vert s_t,a_t]$,我们可以进行蒙特卡洛近似$Q_\pi(s_t,a_t) \approx u_t$，最后我们需要求$u_t$

- 如何求$u_t$？我们玩一局游戏观测到轨迹$(s_t,a_t,r_t,s_{t+1},a_{t+1},r_{t+1},\dots,s_n,a_n,r_n)$，然后计算return:$u_t = \sum_{i=t}^{n}r^{i-t} \cdot r_t$，而且$u_t$是对$Q_\pi(s_t,a_t)$的无偏估计

- 我们还差个$V_{\pi}(s_t)$，我们用神经网络来$V_{\pi}(s_t;\mathbf{w})$近似，于是策略梯度可以近似为：
  $$
  \frac{\partial V_{\pi}(s_t)}{\partial \theta} \approx \mathbf{g}(a_t) \approx \frac{\partial \ln \pi(a_t \vert s_t;\theta) }{\partial \theta} \cdot \left(u_t - v(s_t;\mathbf{w}) \right)
  $$

- 总结下来我们用了三个蒙特卡洛近似：
  $$
  \frac{\partial V_{\pi}(s_t)}{\partial \theta} = \mathbf{g}(A_t) = \mathbb{E}_ {A \sim \pi(\cdot \vert s; \theta)}\left[\frac{\partial  \ln \pi(A \vert s_t;\theta) }{\partial \theta} \ \left(Q_{\pi}(s_t,a_t) -V_\pi(s_t)\right)\right]
  $$
  用$a \sim \pi(\cdot \vert s_t)$去采样动作，这是第一次近似。
  $$
  \mathbf{g}(a_t) = \left[\frac{\partial  \ln \pi(a_t \vert s_t;\theta) }{\partial \theta} \ \left(Q_{\pi}(s_t,a_t) -V_\pi(s_t)\right)\right]
  $$
  然后用$u_t$和$v(s_t;\mathbf{w})$去近似$Q_{\pi}(s_t,a_t) $和$V_\pi(s_t)$，这是第二三次近似：
  $$
  \mathbf{g}(a_t) \approx \frac{\partial \ln \pi(a_t \vert s_t;\theta) }{\partial \theta} \cdot \left(u_t - v(s_t;\mathbf{w}) \right)
  $$

- 这么一来我们就有两个网络了：策略网络$ \pi(a \vert s)$和价值网络：$V(s;\mathbf{w})$，同样地也可以共享feature层的参数。

- 算法步骤

  1. 我们玩一局游戏观测到轨迹$(s_t,a_t,r_t,s_{t+1},a_{t+1},r_{t+1},\dots,s_n,a_n,r_n$
  2. 计算return:$u_t = \sum_{i=t}^{n}r^{i-t} \cdot r_t$ 和 $\delta_t = v(s_t;\mathbf{w}) - u_t$
  3. 更新参数$\theta$和$\mathbf{w}$：
      $$
      \begin{aligned}
      \theta &\gets \theta - \beta \cdot \delta_t \cdot \frac{\partial \ln \pi(a_t \vert s_t;\theta) }{\partial \theta}\\

      \mathbf{w} &\gets \mathbf{w} - \alpha \cdot \delta_t \cdot \frac{\partial v(s_t;\mathbf{w})}{\partial \mathbf{w}}
      \end{aligned}
      $$

#### 4. Policy Gradient 算法的`详细推导`,有点难

在[上面的推导中](#1-具体推导简单版本)，我们说了**假设$Q_\pi$不依赖于$\theta$,但不严谨**，在这里我们写一个完整的推导版本，上面不严谨的推导便于我们快速了解算法，这里是它的真面目。

注意在下面的公式里面 $Q^\pi$和之前出现的$Q_\pi$是一个意义，同样地，$ \pi_\theta(s\vert a)$和 $\pi(s\vert a;\theta)$是一样的意义。

1. 首先我们的目标函数定义为:

    $$
    J(\theta) = \sum_{s\in \mathcal{S}} d^\pi(s) V^\pi(s) = \sum_{s\in \mathcal{S}} d^\pi(s)\sum_{a \in \mathcal{A}} \pi(a\vert s;\theta) Q^{\pi}(s,a)
    $$

    其中 $d^\pi(s)$为策略$\pi(\theta)$决定的马尔科夫平稳分布，其余的符号可以在[术语表中](#术语表)看到。

    马尔科夫链的平稳性：如果我们可以持续永久遍历马尔科夫链中的所有状态，那么随着时间的推移，最终停留在某个状态的概率是一定的，这就是策略 $\pi(\theta)$的平稳性。

    我们定义 $d^\pi(s)$为:

    $d^\pi(s)= \lim_{t \to \infty}P(s_t = s \vert s_0,\pi_\theta)$

    意思是在初始状态 $s_0$开始，遵循策略$\pi$的时候，执行$t$步之后，状态停留在$s$的概率。

    根据目标函数$J(\theta)$的梯度 $\nabla_\theta J(\theta)$，我们可以提升策略梯度算法，最终可以最大化最终收益。

    上面的$J(\theta)$是在连续环境（没有固定的终止状态）下的目标函数(被称为**平均值**)，连续环境下还有一种性质更好的目标函数，叫做**平均回报**：
      $$
      \begin{aligned}
      J(\theta) &\approx r(\pi)\\
      & \approx \lim_{h \to \infty} \frac{1}{h} \sum_{t=1}^h \mathbb{E} [R_t \vert S_0, A_{0:t-1} \sim \pi]\\

      &= \lim_{t \to \infty} \mathbb{E} [R_t \vert S_0, A_{0:t-1}]\\

      &= \sum_s \mu(s) \sum_a \pi_\theta(a \vert s) \sum_{s^{\prime},r} p(s^{\prime},r \vert s,a)r\\

      \end{aligned}
      $$

      我们还定义：
      $$
      \begin{aligned}
      G_t &= R_{t+1} - r(\pi)+R_{t+2} - r(\pi)+R_{t+3} - r(\pi)+\dots\\
      V^\pi(s) &= \sum_a \pi_\theta(a \vert s) \sum_{r,s^\prime} p(s^\prime,r \vert s,a) [r-r(\pi) + V^\pi(s^\prime)]\\

      Q^\pi(s,a)&= \sum_{r,s^\prime} p(s^\prime,r \vert s,a) [r-r(\pi) +\sum_{a^\prime} \pi_\theta(a^\prime \vert s^\prime)Q^\pi({s^\prime,a^\prime})]\\
      \end{aligned}
      $$

      分别为**差分累计回报定义——回报与平均回报差值的累加值、差分状态-价值函数和差分动作状态-价值函数**。

      平均值目标函数是 $J(\theta)$的**另外一种形式**：

      $$
      \begin{aligned}
       J(\theta) &=  \sum_{s\in \mathcal{S}} d^\pi(s) V^\pi(s)\\

      &= \sum_{s\in \mathcal{S}} d^\pi(s)\underbrace {\sum_a \pi_\theta(a \vert s) \sum_{r,s^\prime} p(s^\prime,r \vert s,a) [r +\gamma V^\pi(s^\prime)]}_{V^\pi(s)}\\

      &= r(\pi) + \sum_{s\in \mathcal{S}} d^\pi(s) {\sum_a \pi_\theta(a \vert s) \sum_{r,s^\prime} p(s^\prime,r \vert s,a) [V^\pi(s^\prime)]},根据定义把r(\pi)提出来\\

      &= r(\pi) + \gamma \sum_{s^\prime}V^\pi(s^\prime) \sum_s d^\pi(s) \sum_a \pi_\theta(a \vert s) p(s^\prime \vert s,a)\\

      &= r(\pi) + \gamma \sum_{s^\prime}V^\pi(s^\prime)d^\pi(s^\prime)\\

      &= r(\pi) + \gamma J(\theta)\\

      &= r(\pi) + \gamma (r(\pi) + \gamma J(\theta))\\

      &= \dots\\

      &= \frac{1}{1-\gamma} r(\pi)
      \end{aligned}
      $$

      ***因此在下面的推导中，只考虑平均回报形式的目标函数$r(\pi)$***

      综上所述，目标函数有`3种形式`:
      $$
      \begin{aligned}
      J(\theta) &\overset{\text{def}}{=} \sum_{s\in \mathcal{S}} d^\pi(s)\sum_{a \in \mathcal{A}} Q^{\pi}(s,a) \pi_\theta(a \vert s) \text{连续环境下的平均值形式目标函数,无固定终结状态}\\
      & \overset{\text{def}}{=} \sum_s \mu(s) \sum_a \pi_\theta(a \vert s) \sum_{s^{\prime},r} p(s^{\prime},r \vert s,a)r\text{连续环境下的平均回报形式目标函数,无固定终结状态}\\
      &\overset{\text{def}}{=} V^\pi(s_0) =\mathbb{E}\left[\sum_{t=1}^{+\infty}\gamma^t r_t \vert s_0,\pi\right]\text{周期环境下的目标函数,有固定的起始状态和终结状态}
      \end{aligned}
      $$
2. 策略梯度定理

    由于梯度 $ \nabla_\theta J(\theta)$的计算是非常棘手的，但是我们有策略梯度定理：

    $$
    \begin {aligned}
    \nabla_\theta J(\theta) &= \nabla_\theta \sum_{s\in \mathcal{S}} d^\pi(s) \sum_{a \in \mathcal{A}} Q^{\pi}(s,a) \pi(a \vert s;\theta)\\
    & \propto \sum_{s\in \mathcal{S}} d^\pi(s) \sum_{a \in \mathcal{A}} Q^{\pi}(s,a) \nabla_\theta \pi(a \vert s;\theta)
    \end{aligned}
    $$
3. 策略梯度定理的证明

    首先对状态价值函数求导：

    $$
    \begin {aligned}
    \nabla_\theta V^\pi(s)
    &=\nabla_\theta \left(\sum_{a\in \mathcal{A}} \pi(a\vert s;\theta) Q^\pi(s,a)\right)\\

    &=\sum_{a\in \mathcal{A}} \left(\nabla_\theta \pi_\theta(a\vert s)Q^\pi(s,a) + \pi_\theta(a \vert s) \nabla_\theta Q^\pi(s,a) \right)\\

    &=\sum_{a\in \mathcal{A}}\left(\nabla_\theta \pi_\theta(a\vert s)Q^\pi(s,a) + \pi_\theta(a \vert s) \nabla_\theta \underbrace{\sum_{s^{\prime},r} P(s^\prime,r \vert s,a)(r + V^\pi(s^\prime))}_{Q^\pi(s,a)} \right)\\

    &=\sum_{a\in \mathcal{A}}\left(\nabla_\theta \pi_\theta(a\vert s)Q^\pi(s,a) + \pi_\theta(a \vert s) \sum_{s^{\prime},r} P(s^\prime,r \vert s,a) \nabla_\theta V^\pi(s^\prime) \right) \text{r和状态转移概率与theta无关}\\

    &=\sum_{a\in \mathcal{A}}\left(\nabla_\theta \pi_\theta(a\vert s)Q^\pi(s,a) + \pi_\theta(a \vert s) \sum_{s^{\prime}} \underbrace{P(s^\prime \vert s,a)}_{P(s^\prime \vert s,a) = \sum_{r}P(s^\prime,r \vert s,a)} \nabla_\theta V^\pi(s^\prime) \right)

    \end{aligned}  
    $$

    我们有了：
    $$
    \nabla_\theta V^\pi(s) =\sum_{a\in \mathcal{A}}\left(\nabla_\theta \pi_\theta(a\vert s)Q^\pi(s,a) + \pi_\theta(a \vert s) \sum_{s^{\prime}} P(s^\prime \vert s,a) \nabla_\theta V^\pi(s^\prime) \right)
    $$

    因为上面的等式中同时有 $V^\pi(s)$和 $V^\pi(s^\prime)$,我们可以考虑展开一下。先来做一些准备工作：

    我们定义从状态 $s$ 开始在策略 $\pi_\theta$ 下经过 $k$ 个时间步到达状态 $x$ 的概率记为 $\rho^\pi(s \to x, k)$

      - k=0时，$\rho^\pi(s \to s, 0) = 1 $
      - k=1时，我们遍历在状态$s$时所有可能的动作然后进行一个累加： $ \rho^\pi(s \to s^\prime, k =1 ) = \sum_a \pi_\theta(a \vert s) P(s^\prime \vert s,a)$
      - 如果目标是从状态$s $按照策略$\pi_\theta$经过$k+1$个时间步最终达到了$x$状态，我们可以看做从$s$状态经过$k$个时间步到了 $s^\prime \in \mathcal{S}$ 状态，然后再经过最后一个时间步到达了$x$状态那么我们可以这么计算：

        $\rho^\pi(s \to x,k+1) = \sum_{s^\prime} \rho^\pi(s \to s^\prime,k)\rho^\pi(s^\prime \to x,1)$

    **递归展开$\nabla_\theta V^\pi(s)$**

    为了简洁，我们使用 $\phi(s)$ 来替代 $ \sum_{a \in \mathcal{A}} \nabla_\theta \pi_\theta(a\vert s)Q^\pi(s,a)$

    于是有：
    $$
    \begin{aligned}
    \nabla_\theta V^\pi(s) &=\sum_{a\in \mathcal{A}}\left(\nabla_\theta \pi_\theta(a\vert s)Q^\pi(s,a) + \pi_\theta(a \vert s) \sum_{s^{\prime}} P(s^\prime \vert s,a) \nabla_\theta V^\pi(s^\prime) \right)\\

    &= \phi(s) + \sum_a \pi_\theta(a \vert s)\sum_{s^{\prime}} P(s^\prime \vert s,a) \nabla_\theta V^\pi(s^\prime)\\

    &= \phi(s) + \sum_a\sum_{s^{\prime}} \pi_\theta(a \vert s) P(s^\prime \vert s,a) \nabla_\theta V^\pi(s^\prime)\\

    &= \phi(s)+ \sum_a \underbrace{\rho^\pi(s \to s^\prime, 1 )}
    _
    {
      \sum_{s^{\prime}} \pi_\theta(a \vert s) P(s^\prime \vert s,a)
    } \nabla_\theta V^\pi(s^\prime)\\

    &= \phi(s) + \sum_{s^{\prime}}\rho^\pi(s \to s^\prime, 1 )\left[\phi(s^\prime) + \sum_{s^{\prime\prime}}\rho^\pi(s^\prime \to s^{\prime\prime},1) \nabla_\theta V^\pi(s^{\prime\prime})  \right]\\

    &= \phi(s) + \sum_{s^{\prime}} \rho^\pi(s \to s^\prime, 1 )\phi(s^\prime) + \sum_{s^{\prime}} \rho^\pi(s \to s^\prime, 1 )\sum_{s^{\prime\prime}}\rho^\pi(s^\prime \to s^{\prime\prime},1) \nabla_\theta V^\pi(s^{\prime\prime}) \\

    &= \phi(s) +  \sum_{s^{\prime}} \rho^\pi(s \to s^\prime, 1 )\phi(s^\prime) + \sum_{s^{\prime\prime}}\rho^\pi(s \to s^{\prime\prime},2) \nabla_\theta V^\pi(s^{\prime\prime}) \text{将s撇作为中间状态}\\

    &= \phi(s) +  \sum_{s^{\prime}} \rho^\pi(s \to s^\prime, 1 )\phi(s^\prime) + \sum_{s^{\prime\prime}}\rho^\pi(s \to s^{\prime\prime},2)\phi(s^{\prime\prime}) + \sum_{s^{\prime\prime\prime}}\rho^\pi(s \to s^{\prime\prime\prime},2)\nabla_\theta V^\pi(s^{\prime\prime\prime})\\

    &= \dots \\

    &= \sum_{x\in \mathcal{S}}\sum_{k=0}^{\infty} \rho^\pi(s \to x,k)\phi(x)
    \end{aligned}
    $$

    到此为止我们可以避免计算所谓的 $Q^\pi(s,a)$，然后我们将刚刚推导的式子代入到这里：

    - 在周期环境下的情况

      $$
      \begin{aligned}
      \nabla_\theta J(\theta) &= \nabla_\theta V^\pi(s_0)\\
      &= \sum_s \underbrace{\sum_{k=0}^{\infty} \rho^\pi(s_0 \to s,k)}_{\eta(s)}\phi(s)\\

      &= \sum_s {\eta(s)}\phi(s)\\

      &= (\sum_s \eta(s)) \sum_s \frac{\eta(s)}{\sum_s \eta(s)}\phi(s) \ \text{正则化} \eta(s)\text{使其成为一个概率分布}\\

      &\propto \sum_s \frac{\eta(s)}{\sum_s \eta(s)}\phi(s)\\

      &= \sum_s d^\pi(s) \sum_a \nabla_\theta \pi_\theta(a \vert s) Q^\pi (s,a) \ 因为\sum_s \eta(s)是常数,d^\pi(s)=\frac{\eta(s)}{\sum_s \eta(s)}是平稳分布

      \end{aligned}
      $$

    - 在连续环境下的证明情况，`要用到上面所谓的差分状态-价值函数`

      $$
      \begin{aligned}
      \nabla_\theta V^\pi(s) \\
      &= \nabla_\theta(\sum_{a\in \mathcal{A}} \pi_\theta(a \vert s)Q^\pi(s,a))\\

      &= \phi(s) + \sum_{a\in \mathcal{A}}\pi_\theta(a \vert s)\nabla_\theta Q^\pi(s,a)\\

      &= \phi(s) + \sum_{a\in \mathcal{A}}\left(\pi_\theta(a \vert s)\nabla_\theta \sum_{s^\prime,r} P(s^\prime,r \vert s, a)(r -r(\pi) + V^\pi(s^\prime))\right),展开Q^\pi\\

      &= \phi(s)+ \sum_{a\in \mathcal{A}}\left(\pi_\theta(a \vert s)\left[-\nabla_\theta r(\pi) + \sum_{s^\prime,r} P(s^\prime,r \vert s, a)\nabla_\theta V^\pi(s^\prime) \right]\right),r(\pi)与s^\prime和r无关，可以提出来 \\

      &= \phi(s)+ \sum_{a\in \mathcal{A}}\left(\pi_\theta(a \vert s)\left[-\nabla_\theta r(\pi) + \sum_{s^\prime} P(s^\prime \vert s, a)\nabla_\theta V^\pi(s^\prime) \right]\right) ,消掉一个r，因为P(s^\prime \vert s,a) =\sum_rP(s^\prime,r \vert s, a) \\

      &=-\nabla_\theta r(\pi)+\phi(s) +\sum_{a\in \mathcal{A}}\left(\pi_\theta(a \vert s)\sum_{s^\prime} P(s^\prime \vert s, a)\nabla_\theta V^\pi(s^\prime) \right) \\

      &\Rightarrow  \nabla_\theta r(\pi)= \phi(s)+\sum_{a\in \mathcal{A}}\left(\pi_\theta(a \vert s)\sum_{s^\prime} P(s^\prime \vert s, a)\nabla_\theta V^\pi(s^\prime) \right)-\nabla_\theta V^\pi(s)
      \end{aligned}
      $$

      于是我们平均回报形式的目标函数的梯度有：
      $$
      \begin{aligned}
      \nabla_\theta J(\theta) &= \nabla_\theta r(\pi)\\
      &= \sum_s d^\pi(s) \left[\phi(s)+\sum_{a\in \mathcal{A}}\left(\pi_\theta(a \vert s)\sum_{s^\prime} P(s^\prime \vert s, a)\nabla_\theta V^\pi(s^\prime) \right)-\nabla_\theta V^\pi(s)\right]\\

      &= \sum_s d^\pi(s)\phi(s) + \sum_s d^\pi(s)\sum_{a\in \mathcal{A}}\left(\pi_\theta(a \vert s)\sum_{s^\prime} P(s^\prime \vert s, a)\nabla_\theta V^\pi(s^\prime) \right)- \sum_s d^\pi(s)\nabla_\theta V^\pi(s),因为\sum_s d^\pi(s)=1，所以可以放在前面\\

      &=  \sum_s d^\pi(s)\phi(s) + \sum_{s^\prime}\underbrace{\sum_s d^\pi(s)\sum_{a\in \mathcal{A}}\pi_\theta(a \vert s) P(s^\prime \vert s, a)}_
      {d^\pi(s^\prime)}\nabla_\theta V^\pi(s^\prime) - \sum_s d^\pi(s)\nabla_\theta V^\pi(s)\\

      &= \sum_s d^\pi(s)\phi(s) + \sum_{s^\prime}d^\pi(s^\prime)\nabla_\theta V^\pi(s^\prime)- \sum_s d^\pi(s)\nabla_\theta V^\pi(s)\\

      &= \sum_s d^\pi(s)\phi(s)\\

      &= \sum_s d^\pi(s) \sum_a \nabla_\theta \pi_\theta(a \vert s) Q^\pi (s,a)

      &\propto \sum_s d^\pi(s) \sum_a \nabla_\theta \pi_\theta(a \vert s) Q^\pi (s,a)
      \end{aligned}
      $$
      对于平均值形式的目标函数有：
      $$
      \begin{aligned}
      \nabla_\theta J(\theta) &= \frac{1}{1-\gamma} \nabla_\theta r(\pi)\\
      &= \frac{1}{1-\gamma}\sum_s d^\pi(s) \sum_a \nabla_\theta \pi_\theta(a \vert s) Q^\pi (s,a)\\
      &\propto \sum_s d^\pi(s) \sum_a \nabla_\theta \pi_\theta(a \vert s) Q^\pi (s,a)
      \end{aligned}
      $$
  
    至此，我们有：
      $$
      \begin{aligned}
      \nabla_\theta J(\theta) &\propto \sum_s d^\pi(s) \sum_a \nabla_\theta \pi_\theta(a \vert s) Q^\pi (s,a)\\
      &= \sum_s d^\pi(s) \sum_a \pi_\theta(a \vert s) \frac{\nabla_\theta  \pi_\theta(a \vert s)}{\pi_\theta(a \vert s)} Q^\pi (s,a)\\

      &= \mathbb{E}
      _\pi[Q^\pi (s,a)\nabla
      _\theta \ln\pi
      _\theta(a \vert s) ]
      \end{aligned}
      $$
    也就是上面[简单推导](#1-具体推导简单版本)中的形式了

### 2. Actor Critic On-Policy and Advantage Actor Critic On-policy

直观的来说：使用神经网络来近似价值函数 V，瞎子背着瘸子,仓库中我们实现的是`A2C算法`

- 就目前在网上看到的情况有以下几种 AC 架构

  1. 使用 Actor 来学习策略，Critic 学习 $Q_{\pi}(a,s)$，接受状态 s 作为输入(Policy Based),更新Actor使用`带权重的梯度上升`。
  2. 使用 Actor 来学习策略，Critic 学习 $Q_{\pi}(a,s)$，接受状态 s，a 的 concatenation 作为输入(Value Based)，更新Actor直接使用Critic的输出的`Qvalue`
  3. 与2相同，但是 s 是作为特征（features）从 actor 提取出来的，也就是说共享前面层的参数。

#### 1. AC算法的训练流程

- 训练：

  - 定义：使用神经网络来近似状态-价值函数： $V(s;\theta,\mathbf{w}) = \sum_a \pi(a\vert s;\theta) \cdot q(s,a;\mathbf{w})$.--使用Actor 来学习策略$\pi$ ，Critic 学习动作-状态价值函数$Q_{\pi}(s,a)$
  - 目标：使 policy $\pi(a\vert s;\mathbf{\theta})$ 能够获取最大的回报，$q_{\pi}(s,a;\mathbf{w})$能够更精准的估计动作-状态价值函数
  - 更新 $\theta$ 是为了让 $V(s;\theta,\mathbf{w})$最大，监督完全来自于价值网络-Critic
  - 更新 $\mathbf{w}$ 是为了让 $q_{\pi}(s,a;\mathbf{w})$更加精准，监督完全是来自于环境给的奖励

- 步骤：

  1. 获取状态 $s_t$
  2. 通过 $\pi(\cdot \vert s_t;\mathbf{\theta}_t)$的分布进行一个随机采样，得到下一步的动作$ a_t$
  3. 执行动作，获取状态 $s_{t+1} $和奖励 $r_t$
  4. 使用 td error 来更新 $\mathbf{w}$
      - 计算 $q(s_t,a_t;\mathbf{w})$和 $q(s_{t+1},a_{t+1};\mathbf{w})$
      - 计算 `td target` : $y_t = r_t + \gamma \cdot q(s_{t+1},a_{t+1};\mathbf{w})$ ，显然 $y_t$ 比 $q(s_t,a_t;\mathbf{w})$更加可靠
      - 计算二次距离(也就是`均方误差`):$||y_t - q(s_t,a_t;\mathbf{w})||_2$，
      - 进行梯度下降，让损失变得更小
  5. 使用 policy gradient（策略梯度）来更新 $\mathbf{\theta}$
      - 定义梯度 $g(a,\mathbf{\theta})=\frac{\partial \log \pi(a\vert s;\mathbf{\theta}) }{\partial \mathbf{\theta}} q(s_t,a;\mathbf{w})$，而且上面PG算法中推导了：$\frac{\partial V(s;\mathbf{\theta},\mathbf{w}_t)}{\partial \mathbf{\theta}}=\mathbb{E}_A[\mathbf{g}(A,\mathbf{\theta})]$
      - 由于无法求 $\mathbb{E}_A[\mathbf{g}(A,\mathbf{\theta})]$，我们只能够抽样进行`蒙特卡洛近似`，所以直接使用 $g$ 来代替 $\mathbb{E}_A[\mathbf{g}(A,\mathbf{\theta})]$作为期望的近似,因为$a \sim \pi(\cdot \vert s_t;\mathbf{\theta}_t)$，所以$g$是$\mathbb{E}_A[\mathbf{g}(A,\mathbf{\theta})]$的一个`无偏估计(unbaised estimation)`
      - 进行抽样，并计算 $g(a,\mathbf{\theta}_t)$并进行梯度上升: $\mathbf{\theta}_{t+1} = \mathbf{\theta}_t + \beta \cdot \mathbf{g}(a,\mathbf{\theta}_t)$，使期望越来越高。
      - `注意`：在实际代码中有的时候梯度是 $g(a,\mathbf{\theta})=\frac{\partial \log \pi(a\vert s;\mathbf{\theta}) }{\partial \mathbf{\theta}} q(s_t,a;\mathbf{w})$,有时候是 $g(a,\mathbf{\theta})=\frac{\partial \log \pi(a\vert s;\mathbf{\theta}) }{\partial \mathbf{\theta}} [\underbrace{r_t + \gamma \cdot v(s_{t+1};\mathbf{w})}_{td \ target} - v(s_t;\mathbf{w})]$，前者是不带baseline 后者带baseline，在本仓库中的代码就是后者，它的方差较小，收敛更快，后者的思路可以在下面看到。

- Critic 在训练完毕之后就没有用辣！

#### 2. 使用`baseline`:**Actor Critic With Baseline (A2C)**

- 它也是由两个网络组成：
  
  策略网络(policy network)
  $ \pi(a\vert s;\theta)$和

  状态价值网络(state value function)
  $ v(s;\mathbf{w})$，
  
  状态价值网络是对状态价值函数的近似，它只依赖于状态$s_t$而并不向上面policy gradient算法中$Q_{\pi}(s_t,a_t)$依赖$s_t$和$a_t$，所以它更好训练。

- 训练流程

  - 观测到transition $(s_t,a_t,r_t,s_{t+1})$
  - 计算TD Target $y_t = r_t + \gamma \cdot v(s_{t+1};\mathbf{w})$
  - 计算 TD Error $ \delta_t = v(s_t;\mathbf{w})- y_t$
  - 执行梯度上升更新策略网络$ \pi(a\vert s;\theta)$： $\theta \gets \theta - \beta \cdot \delta_t \cdot \frac{\partial \ln \pi(a_t \vert s_t;\theta)}{\partial \theta}$
  - 执行梯度下降来更新状态价值网络$ v(s;\mathbf{w})$:
  $ \mathbf{w} \gets \mathbf{w} - \alpha \cdot \delta_t \cdot \frac{\partial v(s_t;\mathbf{w})}{\partial \mathbf{w}}$ (使用均方误差作为损失函数)

- 数学推导
  
  - $Q_{\pi}(s_t,a_t) = \mathbb{E}_
  {S_{t+1},A_{t+1}}[R_t+ \gamma \cdot Q_{\pi}(S_{t+1},A_{t+1})]$

  - 于是可以推导出来
    $$
    \begin{aligned}
    Q_{\pi}(s_t,a_t) &= \mathbb{E}_
    {S_{t+1},A_{t+1}}[R_t+ \gamma \cdot Q_{\pi}(S_{t+1},A_{t+1})]\\
    &= \mathbb{E}_
    {S_{t+1}}\left[R_t + \gamma \cdot \mathbb{E}_
    {A_{t+1}}[Q_{\pi}(S_{t+1},A_{t+1})]\right]\\

    &= \mathbb{E}_
    {S_{t+1}}[R_t + \gamma \cdot V_{\pi}(S_{t+1})]
    \end{aligned}
    $$

  - 因为$V_{\pi}(s_t)$的定义为$V_{\pi}(s_t) = \mathbb{E}_{A_{t}}[Q_{\pi}(s_{t},A_{t})]$，把$Q_{\pi}(s_{t},A_{t})$换进去得到
    $$
    \begin{aligned}
    V_{\pi}(s_t) &= \mathbb{E}_{A_{t}}[Q_{\pi}(s_{t},A_{t})]\\
    &= \mathbb{E}_{A_{t}}\left [ \mathbb{E}_
    {S_{t+1}}[R_t + \gamma \cdot V_{\pi}(S_{t+1})]
    \right]\\
    &= \mathbb{E}_{A_{t},S_{t+1}}\left [R_t + \gamma \cdot V_{\pi}(S_{t+1})
    \right]
    \end{aligned}
    $$
  
  - 对$Q_{\pi}(s_t,a_t)=\mathbb{E}_
    {S_{t+1}}[R_t + \gamma \cdot V_{\pi}(S_{t+1})]$做蒙特卡洛近似：

    我们知道了$(s_t,a_t,r_t,s_{t+1})$,那么可以近似$Q_{\pi}(s_t,a_t)$:

    $Q_{\pi}(s_t,a_t)\approx r_t + \gamma \cdot V_{\pi}(s_{t+1})$ (`这是关键公式`)

  - 对$V_{\pi}(s_t)= \mathbb{E}_
  {A_{t},S_{t+1}}\left [R_t + \gamma \cdot V_{\pi}(S_{t+1})
    \right]$做近似

    $V_{\pi}(s_t) \approx r_t + \gamma \cdot V_{\pi}(s_{t+1})$ , `TD target就是这么得来的`。

  - 所以有两个近似:
  
    $Q_{\pi}(s_t,a_t)\approx r_t + \gamma \cdot V_{\pi}(s_{t+1})$

    $V_{\pi}(s_t) \approx r_t + \gamma \cdot V_{\pi}(s_{t+1})$
  
  - 我们在策略梯度里面推导出了

    $$
    \mathbf{g}(a_t) = \left[\frac{\partial  \ln \pi(a_t \vert s_t;\theta) }{\partial \theta} \ \left(Q_{\pi}(s_t,a_t) -V_\pi(s_t)\right)\right]
    $$

    我们称$(Q_{\pi}(s_t,a_t) -V_\pi(s_t))$为`优势函数 Advantage function`这就是A2C比AC多一个A的原因，于是我们用上面的东西做近似：

    $$
    \begin{aligned}
    Q_{\pi}(s_t,a_t) -V_\pi(s_t) &\approx  r_t + \gamma \cdot V_{\pi}(s_{t+1}) - V_\pi(s_t)\\
    &\approx r_t + \gamma \cdot v(s_{t+1};\mathbf{w}) - v(s_t;\mathbf{w}) \  \text{使用神经网络进行近似}
    \end{aligned}
    $$

    把$ r_t + \gamma \cdot v(s_{t+1};\mathbf{w})$ 称为$y_t$

  - 使用近似的策略梯度进行梯度上升，来更新策略网络：
    $$
    \theta \gets \theta + \beta \cdot  \frac{\partial \ln \pi(a_t \vert s_t;\theta)}{\partial \theta}\left(y_t - v(s_t;\mathbf{w})\right)
    $$

  - 使用TD 算法更新价值网络
    $V_{\pi}(s_t) \approx r_t + \gamma \cdot V_{\pi}(s_{t+1})$，用神经网络进行近似得到：
    $v(s_t;\mathbf{w}) \approx r_t + \gamma \cdot v(s_{t+1};\mathbf{w}) $

    TD target $y_t = r_t + \gamma \cdot v(s_{t+1};\mathbf{w})$

    TD算法鼓励$v(s_t;\mathbf{w})$去接近TD target，于是我们将它们相减得到$\delta_t = v(s_t;\mathbf{w}) - y_t$

    执行梯度下降：
    $$ \mathbf{w} \gets \mathbf{w} - \alpha \cdot \delta_t \cdot \frac{\partial v(s_t;\mathbf{w})}{\partial \mathbf{w}}$$

- 对A2C的直观解释
  
  - 我们有近似梯度：
    $$
    \mathbf{g}(a_t)=\frac{\partial \ln \pi(a_t\vert s_t;\theta)}{\partial \theta} \cdot \underbrace{(r_t + \gamma \cdot v(s_{t+1};\mathbf{w})- v(s_t;\mathbf{w}))}_{\text{critic做出的评估}}
    $$

  - $(r_t + \gamma \cdot v(s_{t+1};\mathbf{w})- v(s_t;\mathbf{w}))$如何评价动作的好坏？

    $v(s_t;\mathbf{w})$是$\mathbb{E}[U_t\vert s_t]$的近似，能够评价$s_t$的好坏

    $r_t + \gamma \cdot v(s_{t+1};\mathbf{w})$是在$t+1$时刻对$\mathbb{E}[U_t \vert s_t,s_{t+1}]$的近似，这还是预测，因为$t+1$时刻游戏没有结束，所以$U_t$就是未知的。

    它们都是对$U_t$的预测，但是后者$v(s_t;\mathbf{w})$是在执行$a_t$之前作出来的预测，前者$r_t + \gamma \cdot v(s_{t+1};\mathbf{w})$是在$a_t$执行之后做出的预测，如果$a_t$是好的，那么这个$(r_t + \gamma \cdot v(s_{t+1};\mathbf{w})- v(s_t;\mathbf{w}))$差值就是正的，否则就是负的，所以我们就叫它为`advantage`，意思就是动作$a_t$比平均情况$V_{\pi}(s_t)$好多少呢？

- 与REINFORCE with baseline的异同

  它们都需要策略网络和价值网络，而且`结构没啥区别`。

  但是它们的价值网络的功能是不一样的，A2C的价值网络叫critic，用来评价`动作的好坏`，而REINFORCE 的价值网络就是`仅仅作为baseline`

  两个算法中仅仅就是这个地方不一样：
  $$
  \begin{aligned}
    \mathbf{g}(a_t) &\approx \frac{\partial \ln \pi(a_t\vert s_t;\theta)}{\partial \theta} \cdot \underbrace{(r_t + \gamma \cdot v(s_{t+1};\mathbf{w})- v(s_t;\mathbf{w}))}_{\text{critic做出的评估,这里是单步的TD target}}\\

  \mathbf{g}(a_t) &\approx \frac{\partial \ln \pi(a_t \vert s_t;\theta) }{\partial \theta} \cdot \underbrace{\left(u_t - v(s_t;\mathbf{w}) \right)}_{\text{这里使用的就是极端情况，不使用bootstrapping}}

  \end{aligned}
  $$
- 同样地，我们也可以使用`multi-step TD target` 来提升效果。
  
  从这个角度来看`REINFORCE 是特殊的multi-step A2C`，它用了所有的奖励，不做bootstrapping。

### 3. DDPG Off-Policy

全称是**Deep Deterministic Policy Gradient**，意思就是它的策略是`确定性`的。

#### 1. DDPG的特征

![ddpg algo](./DeepDeterministicPolicyGradient(DDPG)/principle.png)

- Exploration noise ($\mu^\prime(s) = \mu_\theta(s) + \mathcal{N}$)
- Actor-Critic Achetecture
- Fixed Q-Target
- Policy Gradient
- Experience Replay (OFF-POLICY)
- Soft Update (Conservative strategy iteration)

#### 2.确定性策略梯度的推导

上面的方法中我们策略函数$\pi(\cdot \vert s)$总是描述成给定状态$s$下载动作空间$ \mathcal{A}$上的概率分布，所以策略是`随机的`，在确定性策略梯度(Deterministic Policy Gradient)中，我们将环境建模为一个`确定性的决策`：$ a = \mu(s)$

之前我们有：

- $\rho_0(s)$ 初始状态分布
- $\rho^\mu(s \to s^\prime ,k) $ 从状态$s$开始，遵循策略$\mu$经过$k$个时间步到达状态$ s^\prime$的访问概率密度

- $\rho^\mu(s^\prime)$，折扣状态分布，定义为$\rho^\mu(s^\prime)= \int_{\mathcal{S}} \sum_{k=1}^{\infty} \gamma^{k-1}\rho_0(s) \rho^\mu(s \to s^\prime,k) ds$

于是平均值形式的目标函数定义为：
$$
J(\theta) = \int_{\mathcal{S}} \rho^\mu(s)Q(s,\mu_\theta(s))ds
$$

个人理解为对于所有的状态$s$我们计算它出现的概率，然后乘以一个根据策略$\mu(s)$做出来的策略的$Q$值，然后我们对$s$求个积分，也就是求平均值。

确定性策略梯度：
$$
\begin{aligned}
\nabla_\theta J(\theta) &= \int_\mathcal{S} \rho^\mu(s) \nabla_a Q^\mu(s,a) \nabla_\theta \mu_\theta(s) \vert_{a=\mu_\theta(s)} ds\\
&= \mathbb{E}_{s \sim \rho^\mu}[\nabla_a Q^\mu(s,a) \nabla_\theta \mu_\theta(s) \vert_{a=\mu_\theta(s)}]
\end{aligned}
$$

问题在于，我们如果遵循确定性的策略，除非环境有很多噪声，我们很难进行充分的探索，所以我们可以在确定策略中添加噪声或者遵循不同的随机行为策略来进行样本的收集，以便离线学习。

在离线方法中，训练的样本是通过一个随机行为策略$\beta(a \vert s)$产生的，因此状态分布服从于对应的折扣状态密度$\rho^\beta$：

$$
\begin{aligned}
J_\beta(\theta) &= \int_{\mathcal{S}} \rho^\beta Q^\mu(s,\mu_\theta(s))ds\\

\nabla_\theta J_\beta(\theta) &= \mathbb{E}_{s \sim \rho^\beta}[\nabla_a Q^\mu(s,a) \nabla_\theta \mu_\theta(s) \vert_{a=\mu_\theta(s)}]

\end{aligned}
$$

直观地来说，使用$\mu(s)$策略来进行经验收集可能会导致环境探索不充分，进而导致 $Q$ value收敛不充分，我们的策略就不完美。所以一般可以用另外的随机策略进行经验收集。

注意：*在本仓库的DDPG算法中，以上两个手段都用到了，具体的可以看代码*

### 4. A3C On-Policy

AC的`多线程`版本

#### 1. A3C的特征

- A3C 里面有多个 agent 对网络进行异步更新，相关性较低
- 不需要积累经验，占用内存少
- on-policy 训练
- 多线程异步,速度快
- 注意：A3C与MARL多智能体强化学习不同 A3C着重与并行训练，梯度累积可以认为是minibatch SGD

### 5. PPO On-Policy

#### 1.PPO的特征

- 使用 importance sampling 来使用过去的经验
- PPO 是积累部分经验(一个 trajectory)，然后进行多轮的梯度下降
- 对 importance weight 进行裁剪从而控制更新步长

#### 2. PPO的细节

鉴于TRPO很复杂但是我们仍然想模仿一下TRPO的思想，近端策略优化(proximal policy optimization)就是通过使用一个截断的替代目标函数来简化TRPO。

我们令TRPO中的importance sampling weight 为:
$$
r(\theta) = \frac{\pi_\theta(a \vert s)}{\pi_{\theta_{old}}(a \vert s)}
$$

仿照TRPO，我们的目标函数(在线的情况)为：
$$
J^{TRPO}(\theta)=\mathbb{E}[r(\theta) \hat A_{\theta_{old}}(s,a)]
$$

在TRPO的更新过程中，我们需要保证$\theta_{old}$和$\theta$的距离，否则会导致参数过大幅度的变动使训练过程不稳定。

在PPO中，强行使得$r(\theta)$保持在1附近的领域中即$[1-\epsilon,1+\epsilon]$，来保持这一约束，在本仓库中$\epsilon = 0.2$是超参数。

$$
J^{CLIP}(\theta) = \mathbb{E}\left[ \min \left(r(\theta)\hat A_{\theta_{old}}(s,a),\text{clip}(r(\theta),1-\epsilon,1+\epsilon)\hat A_{\theta_{old}}(s,a)\right) \right]
$$

clip函数就是将$r(\theta)$截断在$[1-\epsilon,1+\epsilon]$范围内，这里违背了TRPO的一个理念：尽可能地最大化策略的更新幅度从而得到更好的回报。

实际应用在Actor Critic共享参数的网络结构上面的时候，我们的目标函数还加上了关于值估计的误差项还有一个熵正则项(鼓励探索)。

$$
J^{{CLIP}^\prime}(\theta) = \mathbb{E}[J^{CLIP}(\theta) + c_1 (V_\theta(s) - V_{target})^2 + c2 \mathcal{H}(s,\pi_\theta(\cdot))]
$$

其中$c_1,c_2$为两个超参数。

个人觉得PPO的优点就是收敛快，对于LunarLander来说，它只要15分钟就可以收敛，缺点是使用$\theta_{old}$会导致环境探索不充分进而导致停留在局部最小。

### 6. TRPO On-Policy

#### 1. TRPO的特征

- 使用 $\mathcal{L}(\theta|\theta_{old})$来近似目标函数 $J(\theta)$
- 使用 KL 散度或者是二次距离来约束 $\theta$ 与 $\theta_{old}$ 之间的差距
- 因此，相比于普通的 PG 算法，它更稳定，因为他对于学习率不敏感

#### 2. 离线策略梯度的一些证明

注意：这个证明是给使用`带权重的梯度上升`算法的`离线学习`用的，DDPG不是，因为它还是使用Qvalue作为损失函数。

离线方法有以下优势：

  1. 离线算法不需要完整的轨迹样本，并且可以复用任何历史轨迹的样本(Experience Replay 经验回放)从而实现了Sample Efficiency，样本有效性。
  2. 训练样本根据行为策略而不是目标策略收集的，给算法带来了更好的探索性。

有以下定义：

  1. $\beta(a \vert s)$ 行为策略，用来收集训练样本
  2. $\pi_\theta(a \vert s)$ 目标策略，我们要训练的策略
  3. $d^\beta (s)$ 根据行为策略$\beta$导出的平稳分布，$d^\beta (s)= \lim_{t \to \infty} P(S_t=s\vert S_0,\beta)$
  4. $Q^\pi$ 根据目标策略$\pi$估计的action-value function

我们的平均值形式的目标函数有：

$$
J(\theta) = \sum_{s\in \mathcal{S}} d^\beta (s) \sum_{a\in \mathcal{A}} Q^\pi(s,a)\pi_\theta(a \vert s) = \mathbb{E}_{s \sim d^\beta}\left[\sum_{a \in \mathcal{A}} Q^\pi(s,a)\pi_\theta(a\vert s)\right]
$$

于是我们可以改写成以下形式：
$$
\begin{aligned}
\nabla_\theta J(\theta) &= \nabla_\theta \mathbb{E}
_
{s \sim d^\beta} \left[\sum_{a \in \mathcal{A}} Q^\pi(s,a)\pi_\theta(a\vert s)\right]\\

&= \mathbb{E} _
{s \sim d^\beta}\left[\sum
_ {a \in \mathcal{A}} (Q^\pi(s,a)
\nabla_\theta\pi
_ \theta(a \vert s) +
\nabla_\theta Q^\pi(s,a)\pi
_\theta(a \vert s))\right]\\

& \approx \mathbb{E} _
{s \sim d^\beta}\left[\sum
_ {a \in \mathcal{A}} Q^\pi(s,a)
\nabla_\theta\pi
_ \theta(a \vert s) \right](忽略掉后面一项，进行一下近似)\\

&= \mathbb{E}
_
{s \sim d^\beta}\left[\sum
_ {a \in \mathcal{A}} \beta(a\vert s) \frac{\pi_\theta(a\vert s)}{\beta(a\vert s)}Q^\pi(s,a)
\frac{\nabla_\theta\pi
_
\theta(a \vert s)}{\pi_\theta(a\vert s)} \right]\\
&= \mathbb{E}_
\beta\left[\frac{\pi
_
\theta(a\vert s)}{\beta(a\vert s)}Q^\pi(s,a)
\nabla \ln \pi_\theta(a\vert s)\right]

\end{aligned}
$$
我们称$\frac{\pi_\theta(a\vert s)}{\beta(a\vert s)}$为重要性权重(importance weight)

在这里我们忽略掉了$\nabla_\theta Q^\pi(s,a)$，使用近似的策略梯度，这样依旧能够保证使用梯度上升算法能够使得的略性能提升并收敛到一个真实的局部最优解。

#### 3.TRPO的具体细节

为了提升训练的稳定性，我们应该避免更新一步就使得策略发生剧烈变化的参数更新，为此我们标出一个置信区间，参数在这个区间更新就是安全的，所以这个方法叫做 Trust Region Policy Optimization，它是通过在每次迭代的时候对策略更新的幅度强制施加KL_divergence约束来实现上述理念。

对于离线情况，目标函数的衡量是遵循一个不同的行为策略$\beta(a \vert s)$来进行采样的同时，在状态访问分布以及动作空间上的所有优势：
$$
\begin{aligned}
J(\theta) &= \sum_{s\in \mathcal{S}} \rho^{\pi_{old}} \sum_{a \in \mathcal{A}} \left (\pi_\theta(a \vert s) \hat A_{\pi_{old}}(s,a) \right)\\

&= \sum_{s\in \mathcal{S}} \rho^{\pi_{old}} \sum_{a \in \mathcal{A}} \left (\beta(a\vert s)\frac{\pi
_
\theta(a\vert s)}{\beta(a\vert s)} \hat A_{\pi_{old}}(s,a) \right)\\

&= \mathbb{E}_
{s \sim \rho^{\pi
_
{\theta_
{old}}},a\sim \beta} \left[\frac{\pi
_
\theta(a\vert s)}{\beta(a\vert s)} \hat A_{\pi_{old}}(s,a)\right]
\end{aligned}
$$

其中$\pi_{old}$是我们已知的，是更新之前的策略参数，$\beta(a\vert s)$是用来采样轨迹的行为策略，我们使用的$\hat A(\cdot)$是对于真实优势函数$A(\cdot)$的一个蒙特卡洛估计。上面的公式是我从那个 [桃子网站](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/) 抄下来的，个人觉得如果是使用$\beta$来进行采样的话，那么第一个$\sum$符号下标应该就是$\rho^\beta$，这应该是**有错误的**。

对于在线情况，我们的行为策略是$\pi_{\theta_{old}}(a \vert s)$:

$$
J(\theta) = \mathbb{E}_
{s \sim \rho^{\pi
_
{\theta_
{old}}},a\sim \pi_{\theta_{old}}} \left[\frac{\pi
_
\theta(a\vert s)}{\pi_{\theta_{old}}(a\vert s)} \hat A_{\pi_{old}}(s,a)\right]
$$

TRPO算法旨在`满足置信区间的约束下`最大化目标函数$J(\theta)$,该约束强制就策略与新策略之间的KL散度小于某个参数$\delta$：

$$
\mathbb{E}_{s \sim \rho^{\pi_{\theta_{old}}}}[D_{KL}(\pi_{\theta_{old}}(\cdot \vert s) \vert \pi_\theta(\cdot \vert s))] \le \delta
$$

### 7. Soft Actor Critic Off-Policy (实现了 PER)

CODE：[SAC](<./SoftActorCritic(SAC)/SoftActorCritic>)

#### 1. SAC的特征

1. 采用分离策略网络以及值函数网络的 **AC 架构**，这里 actor 是学习策略，使 Q 值最大（即 state-action value 最大）
2. ER 能够使用历史数据，高效采样
3. 熵最大化以鼓励探索
4. 采用 target net 和 double critic 架构
5. reparameterize 使 log standard deviation 可微
6. 一次采样多次进行梯度下降

#### 2. SAC的细节

SAC的目标是`同时最大化期望累计回报和策略的熵度量`：

$$
J(\theta) = \sum_{t=1}^{T} \mathbb{E}_{(s_t,a_t) \sim \rho_{\pi_\theta}}[r(s_t,a_t) + \alpha \mathcal{H}(\pi_\theta(\cdot \vert s))]
$$

其中$\mathcal{H}(\cdot)$表示熵度量,$\alpha$表示热度，用于控制熵度量的比例。

熵最大化可以使策略在训练过程中可以：1.进行更多的探索， 2.捕获近似最优策略的多种模式（同等好的策略应该有一样的概率被选中）。

SAC旨在学习三个函数：

- 策略$\pi_\theta$，参数是$\theta$
- soft Q Value funtion $Q_\omega$，参数是$\omega$
- soft State-Value funtion $V_\psi$，参数是$\psi$,理论上可以用策略$\pi$和$Q$来推导出$V$,在实际情况下显式对状态价值函数建模可以使得训练过程更加稳定

在SAC中，soft Q-value 和 Soft State-value 分别定义如下：
$$
\begin{aligned}
Q(s_t,a_t) &= r(s_t,a_t) + \gamma \cdot \mathbb{E}_
{s_{t+1} \sim \rho_\pi(s)}[V(s
_{t+1})]\\

V(s_t) &= \mathbb{E}_{a_t \sim \pi}[Q(s_t,a_t) - \alpha \log \pi(a_t \vert s_t)]\\

\end{aligned}
$$

于是我们可以根据上面两个公式推导出：
$$
Q(s_t,a_t) = r(s_t,a_t) + \gamma \cdot \mathbb{E}_
{(s_{t+1},a_{t+1}) \sim \rho_\pi}[Q(s_{t+1},a_{t+1}) - \alpha \log \pi(a_{t+1} \vert s_{t+1})]
$$

其中 $\rho_{\pi}(s)$和 $\rho_{\pi}(s,a)$是由策略$\pi(a\vert s)$导出的状态分布的状态以及状态-动作边际分布。

soft state value function使用MSE来训练,其中$\mathcal{D}$代表经验池：
$$
J_V(\psi) = \mathbb{E}
_
{s_t \sim \mathcal{D}}\left[\frac {1}{2} \left(V_\psi(s_t) - \mathbb{E}[Q_\omega(s_t,a_t)-\log \pi_\theta(a_t \vert s_t)]\right)^2\right]\\

\nabla_\psi J_V(\psi) = \nabla_\psi V_\psi\left(V_\psi(s_t) - \mathbb{E}[Q_\omega(s_t,a_t)-\log \pi_\theta(a_t \vert s_t)]\right)
$$

soft Q value function 通过最小化软贝尔曼残差来训练：
$$
J_Q(\omega) = \mathbb{E}
_
{(s_t,a_t) \sim \mathcal{D}}\left[\frac{1}{2} (Q_\omega(s_t,a_t)-(r(s_t,a_t)+ \gamma \mathbb{E}
_
{s
_
{t+1} \sim \rho_\pi(s)}[V_{\bar\psi}(s_{t+1})]))\right]\\

\nabla_\omega J_Q(\omega) = \nabla_\omega Q_\omega(s_t,a_t)(Q_\omega(s_t,a_t) -r(s_t,a_t)-\gamma V_{\bar\psi}(s_{t+1}))
$$
其中$ \bar \psi$是target value function，它是一个指数移动平均值，像DQN中的target network一样进行soft update。

然后SAC策略更新的目标是尝试去最小化KL散度(KL-divergence):
$$
\begin{aligned}
\pi_{new} &= \arg \min_{\pi^\prime \in \prod} D_{KL }\left(\pi^\prime(\cdot \vert s_t) \vert \vert \frac{\exp (Q^{\pi_{old}}(s_t,\cdot))}{Z^{\pi_{old}}(s_t)}\right)\\

&=\arg \min_{\pi^\prime \in \prod} D_{KL }\left(\pi^\prime(\cdot \vert s_t) \vert \vert \exp(Q^{\pi_{old}}(s_t,\cdot)-\log Z^{\pi_{old}}(s_t))\right)
\end{aligned}
$$

SAC策略网络的目标函数为：
$$
\begin{aligned}
J_\pi(\theta) &= \nabla_\theta D_{KL} (\pi_\theta(\cdot \vert s)\vert \vert \exp(Q_{\omega}(s_t,\cdot)-\log Z_{\omega}(s_t)))\\
&= \mathbb{E}_
{a_t \sim \pi} \left[- \log \left(\frac{\exp(Q_{\omega}(s_t,\cdot)-\log Z_{\omega}(s_t))}{\pi_\theta(a_t \vert s_t)}\right)\right]\\

&= \mathbb{E}_
{a_t \sim \pi} \left[\log \pi_\theta(a_t \vert s_t) - Q_\omega(s_t,a_t) + \log Z_\omega(s_t) \right]
\end{aligned}
$$
其中$\prod$是潜在策略的集合，我们可以将这些策略建模为更容易处理的形式，例如$\prod$可以是高斯混合分布族。 $Z^{\pi_{old}}(s_t)$是用于正则化分布的配分函数。最小化$J_\pi(\theta)$的方式依赖于$\prod$的选择。

更新方式能够保证$ Q^{\pi_{new}}(s_t,a_t) \ge Q^{\pi_{old}}(s_t,a_t) $

算法具体步骤：

  1. 初始化$ \theta,\omega,\psi,\bar \psi$。
  2. 对于每轮循环，执行：
  3. （在实际操作中，一次与环境交互，采取多次梯度下降是最好的）
  4. 对于每轮环境，执行：
  5. $ a_t \sim \pi_\theta(a_t \vert s_t)$
  6. $s_{t+1} \sim \rho_\pi(s_{t+1} \vert s_t,a_t)$
  7. $\mathcal{D} \gets \mathcal{D} \cup \mathcal{D}{(s_t,a_t,r(s_t,a_t) ,s_{t+1})} $
  8. 对于每轮的梯度下降步骤，执行：
  9. $\psi \gets \psi - \lambda_V \nabla_\psi J_V(\psi)$
  10. $\omega \gets \omega - \lambda_Q \nabla_\omega J_Q(\omega)$
  11. $\theta \gets \theta - \lambda_\pi \nabla_\theta J_\pi(\theta)$
  12. $\bar \psi \gets \tau \psi + (1-\tau) \hat \psi$

之后还出现了带有自动热度调整的SAC算法，在代码库中这两种都实现了。

### 8. TwinDelayedDeepDeterministicPolicyGradient(TD3) Off-Policy

#### 1. TD3 的特征

其实就是结合了许多之前的优点，还能改进的话就加熵或者是在经验池上面做文章。

1. 双 Critic
2. 延迟更新 Actor
3. soft update
4. 使用 replay buffer

#### 2. TD3的具体细节

Qlearning的一个缺点就是对于值函数的过估计，这个会通过bootstrapping进一步扩大过估计，给策略学习造成负面的影响。

TD3在DDPG算法的基础上进一步改进，防止值函数的过估计。使用两个值网络将动作选择和Q值更新解耦。

1. 截断双Q学习：

    在双Q学习中，动作选择以及Q值估计是通过两个独立的网络完成的。在DDPG中，给定两个确定性的策略网络$ (\mu_{\theta_1} \mu_{\theta_2})$和对应的两个Critic $(Q_{\omega_1},Q_{\omega_2})$ 于是双Q学习的bellman目标如下：

    $$
    \begin{aligned}
    y_1 &= r + \gamma Q_{\omega_2}(s^\prime ,\mu_{\theta_1}(s^\prime))\\
    y_2 &= r + \gamma Q_{\omega_1}(s^\prime ,\mu_{\theta_2}(s^\prime))
    \end{aligned}
    $$

    由于策略变化过于缓慢，两个Actor会过于相似，从而很难作出完全独立的决策，与是截断双Q学习使用两者之间最小的估计，从而倾向于使用难以通过训练传播的欠估计误差：

    $$
    \begin{aligned}
    y_1 &= r + \gamma \min_{i=1,2}Q_{\omega_i}(s^\prime ,\mu_{\theta_1}(s^\prime))\\
    y_2 &= r + \gamma \min_{i=1,2}Q_{\omega_i}(s^\prime ,\mu_{\theta_2}(s^\prime))
    \end{aligned}
    $$

2. 延迟更新目标和策略网络

    在AC架构中，策略与值函数的更新深度耦合，当策略较差的时候值函数的估计会由于过估计发散，如果值函数估计不准又会使策略变差。

    为了减小方差，我们更新策略的频率会低于Q值函数，我们的思想是：先等Critic拟合之后再更新Actor。这个想法类似于定期更新目标网络思想在DQN中的使用。

3. 目标策略的平滑

    确定性策略会过拟合到值函数的峰值上面，TD3在值函数上面引入了平滑正则化策略，在所选的动作中添加少量经过截断的随机噪声，并对小批量数据进行平均来减少方差:

    $$
    y = r + \gamma Q_{\omega} (s^\prime, \mu_\theta(s^\prime) +\epsilon)\\

    \epsilon \sim clip (\mathcal{N}(0,\sigma) ,-c ,+c)
    $$

### 9. Actor Critic with Experience Replay (ACER) Off-Policy

#### 1. ACER的特征

1. 使用Retrace Q Estimation
2. 使用偏差校正截断重要性权重
3. 更高效的TRPO

#### 2. 具体算法

1. Retrace Q值估计
Retrace 是一种离线的基于累计回报的Q值估计算法，它对任意的目标-策略网络$(\pi,\beta)$都有一个比较好的收敛性保证并且拥有很好的数据有效性。

    我们TD学习计算误差是这样的:
    $$
    \delta_t = R_t + \gamma \mathbb{E}_{a \sim \pi}Q(S_{t+1},a) - Q(S_t,A_t)
    $$

    其中$R_t + \gamma \mathbb{E}
    _
    {a \sim \pi}Q(S
    _
    {t+1},a)$的观测值$r_t + \gamma \mathbb{E}_{a \sim \pi}Q(s
    _
    {t+1},a)$被称为TD traget ，使用期望值是因为如果我们遵循大年策略$\pi$的时候对于未来时间步我们能够做的最好估计也就是累计回报可能是多少。

2. 通过校正误差往目标移动来更新Q值$ Q(S_t,A_t) \gets Q(S_t,A_t) + \alpha \delta_t $所以我们的Q值的更新幅度是与$\delta_t$高度相关的。

    在离线学习中，我们需要对Q值进行重要性采样：

    $$\alpha \delta_t = \Delta Q(S_t,A_t)$$
    $$ \Delta Q^{imp}(S_t,A_t) = \gamma^t \prod_{1 \le \tau \le t} \frac{\pi(A_\tau \vert S_\tau)}{\beta(A_\tau \vert S_\tau)}\delta_t $$

    中间连乘符号带来了很大的方差，而且连乘会导致这个$\delta_t$过大或者过小，过小大不了就是训练慢了，过大会导致不稳定。

    所以我们需要截断一下这个Q值：
    $$
    \Delta Q^{ret} (S_t, A_t) = \gamma^t \prod_{1 \le \tau \le t}\min \left( \frac{\pi(A_\tau \vert S_\tau)}{\beta(A_\tau \vert S_\tau)},c\right)\delta_t
    $$

    还是通过最小化L2 Loss来训练Critic $[Q^{ret} (s,a) - Q(s,a)]^2$

3. 重要性权重截断

    为了减少估计策略梯度$\hat g$产生的高方差，ACER使用一个常数加一个矫正项来截断Importance Weight重要性权重，$\hat g^{acer}$代表$t$时候对于策略梯度的一个估计，我们令$ \omega_t =\frac{\pi(A_t \vert S_t)}{\beta(A_t \vert S_t)},\omega_t(a) =\frac{\pi(a_t \vert S_t)}{\beta(a_t \vert S_t)} $：

    $$
    \begin{aligned}
    \hat g^{acer} &= \omega_t (Q^{ret}(S_t,A_t)-V_{\theta_v}(S_t)) \nabla_ \theta \ln \pi_\theta(A_t \vert S_t)\\
    &= \min(c,\omega_t)(Q^{ret}(S_t,A_t)-V_{\omega}(S_t)) \nabla_ \theta \ln \pi_\theta(A_t \vert S_t)+\\ & \underbrace{\mathbb{E}_{a \sim \pi} \left [\max(0,\frac{\omega_t(a) -c}{\omega_t(a)}) (Q_\omega(S_t,A_t)-V_{\omega}(S_t)) \nabla_ \theta \ln \pi_\theta(A_t \vert S_t)\right]}_{校正项}
    \end{aligned}
    $$

    其中$Q_\omega(\cdot)$以及$V_\omega(\cdot)$是由$\omega$参数化的critic预测的动作状态价值以及状态价值，加号左边的一项包含了截断重要性权重，第二项是校正项，使得上述方差估计为无偏估计。

4. 高效TRPO

    ACER不再去计算当前策略与更新一步之后的新策略之间的KL divergence，而是维护一个历史策略的运行平均(Running Average)并且强制新策略不会偏离这个平均策略太远。

### 10. Multi-agent DDPG Off-Policy

[MADDPG 论文](https://arxiv.org/pdf/1706.02275.pdf)

[参考](https://tomaxent.com/2019/04/14/%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6%E6%96%B9%E6%B3%95/)

#### 1. MADDPG的特征

- 中心化Critic+ 去中心化Actor
- Agent能够使用估计的其它Agent的策略进行学习
- 策略集成(policy ensembing)能够减少方差

#### 2. MADDPG的具体细节

MADDPG将DDPG扩展到多智能体的环境当中，其中包含多个智能体在仅仅依靠局部信息的情况下进行协作完成任务，从单个智能体的角度来看，环境是非平稳的，因为其它智能体的策略在很快地更新并且一直是未知的。MADDPG是一个经过重新设计的ActorCriti算法，专门用于处理这种不断变化的环境以及智能体的互动。

这个问题可以建模为多智能体斑斑的MDP，也被称为马尔科夫游戏，其中共有$N$个智能体，公共的状态空间为$\mathcal{S}$，每个智能体都有自己的动作空间$\mathcal{A}_1,\dots,\mathcal{A}_N$,以及观察空间$\mathcal{O}_1,\dots,\mathcal{O}_N$，状态转移函数包括所有的状态、动作以及观察空间：$\mathcal{T}: \mathcal{S}\times \mathcal{A}_1 \times \dots \mathcal{A}_N \mapsto \mathcal{S} $,每个智能体自己的随机策略仅仅用到属于自己的观察以及动作:$\pi_{\theta_i} : \mathcal{O}_i \times \mathcal{A}_i \mapsto [0,1]$,或者是一个确定性的策略：
$\mu_{\theta_i}:\mathcal{O}_i  \mapsto \mathcal{A}_i$

我们令$ \vec o = o_1,\dots,o_N, \vec \mu = \mu_1,\dots,\mu_N$并且策略是由$\vec \theta = \theta_1, \dots, \theta_N$初始化的。

MADDPG中的Critic为第$i$个智能体学习一个中心化的动作-值函数$ Q_i^{\vec \mu}(\vec o,a_1,\dots,a_N)$，其中$a_1 \in \mathcal{A}_1,\dots a_N \in \mathcal{A}_N $是每个智能体的动作，每一个$Q_i^{\vec \mu}, i=1,\dots ,N$都是独立学习的,因此每个智能体可以拥有任意形式的回报函数，包括竞争环境中相互冲突的回报函数。同时每个智能体的Actor也是独立探索以及独立更新参数$\theta$。

Actor更新:

$$
\nabla_{\theta_i} J(\theta_i) = \mathbb{E}_{\vec a, a\sim D}\left[\nabla_{a_i} Q_i^{\vec \mu} (\vec o,a_1,\dots,a_N)\nabla_{\theta_i}\mu_{\theta_i}(o_i) \vert_ {a_i = \mu_{\theta_i}(o_i)}\right]
$$

Critic更新:

$$
\mathcal{L}_{\theta_i} = \mathbb{E}_{\vec o,a_1,\dots,a_N,r_1,\dots,r_N,\vec{o ^\prime}}[(Q_i^{\vec \mu} (\vec o,a_1,\dots,a_N) - y)^2]\\

y = \underbrace{r_i + \gamma Q_i^{\vec {\mu^\prime}} (\vec o^\prime,{a_1^\prime},\dots,a^\prime_N) \vert_ {a_j^\prime = \mu_{\theta_j}^\prime}}_{TD \ Target}
$$

其中$\vec \mu^\prime$是延迟软更新参数的目标策略

为了缓解多智能体合作或者竞争带来的高方差，MADDPG提出了策略集成：

1. 一个Agent训练K个策略
2. 随机选取一个策略用于轨迹采样(OPENAI称之为Rollout)
3. 使用K个策略的集成梯度来进行参数更新

### 11. 总结

- DQN、Qlearning、Sarsa 等都在学习状态或者行为价值函数，然后再根据价值函数来选择未来的行为，而策略梯度直接学习策略本身
- 策略梯度方法主要特点在于直接对策略进行建模，通常建模为由 theta 参数化的函数 $\pi_\theta(a \vert s)$，回报函数的值收到该策略的直接影响，于是我们可以用多种方法来最大化回报函数
- Actor-Critic：学习策略和价值函数
- Asynchronous Advantage Actor Critic：侧重于并行训练
- Advantage Actor Critic：引入协调器，收敛更快，性能比 A3C 更好
- Deterministic Policy Gradient：将环境建模为一个确定性的决策：$a=\mu(s)$
- Deep Deterministic Policy Gradient:结合了 DPG 和 DQN 的 AC 架构，DDPG 算法在学习一个确定性策略的同时通过Actor Critic框架将其扩展到连续的动作空间中
- Trust Region Policy Optimization：为了提升训练的稳定性，我们应该避免更新一步就使得策略发生剧烈变化的参数更新。置信区间策略优化通过在每次迭代时对策略更新的幅度强制施加 KL 散度约束来实现上述理念。
- Proximal Policy Optimization：实现了 TRPO 的性能，通过使用一个截断的替代目标来简化算法
- Actor Critic with Experience Replay:离线的 A3C 算法，使用了一系列操作来克服离线算法的不稳定性
- Soft Actor Critic：将策略的熵度量纳入回报函数中用以鼓励探索：我们希望学习到一种尽可能随机行动的策略，同时仍然能够在任务中完成目标。它是一个遵循最大熵强化学习框架的离线演员-评论家模型。一个先例工作是软 Q 学习。
- Twin Delayed Deep Deterministic:在 DDPG 算法的基础上应用了很多新的改进从而防止值函数的过估计现象
- CONCLUSION
  - 尽量减少方差并保持偏差来稳定训练过程
  - 使用离线方法来保持高探索度
  - 使用经验回放来提高效率
  - 可以学习确定性的策略（deterministic）
  - 避免对值函数的过度估计（over estimation）
  - 使用目标网络，目标网络更新较慢(周期性更新或者是软更新)
  - 使用批量归一化(batch norm)
  - 使用熵规范化后的奖励
  - Actor和Critic在前几层共享参数，但是需要仔细考虑是否要共享
  - 在策略更新的过程中保持KL散度约束
  - 使用新的策略优化方法(K-FAC)
  - 使用熵最大化来鼓励探索

---

## 4. Requirements

本仓库使用pipreqs ./ --encoding=utf8生成requirements.txt

[requirements.txt there, run pip install -r requirements.txt](./requirements.txt)

- box2d (box2d for lunarlander-v2 and other gym envs, download WHL file and execute "pip install \*\*\*.whl" otherwise you will suffer building problems o(╥﹏╥)o)
- gym==0.21.0 (Incorrect versions of the gym environment can cause errors, such as v1 and v2 of LunarLander and v0 and v1 of pendulum)
- ipython==7.31.0 (jupyter notebook)
- matplotlib==3.4.3 (jupyter notebook)
- numpy==1.20.3
- pandas==1.3.4
- PyYAML==6.0 (In some algorithms such as SAC, TD3, the settings are stored in a YAML file and need to be read with a library)
- tensorboardX==2.4.1
- torch==1.10.1+cu113
- torchvision==0.11.2+cu113
- typing_extensions==3.10.0.2

---

## 5. 杂谈&经验

### 1. 关于Pytorch和Python

- Tensor.to(device)操作要细心，有可能梯度为 None 因为.to(device)是一次操作，之后的 tensor 有一个 grad_fn=copy 什么的，此时的 tensor 不再是叶子结点。
- nn.parameter()通常，我们的参数都是一些常见的结构（卷积、全连接等）里面的计算参数。而当我们的网络有一些其他的设计时，会需要一些额外的参数同样很着整个网络的训练进行学习更新，最后得到最优的值，经典的例子有注意力机制中的权重参数、Vision Transformer 中的 class token 和 positional embedding 等。
- tensor.clone()=原来张量的拷贝，而且 require_grad=True
- t.tensor.detach()： 返回 t.tensor 的数据而且 require\*grad=False.torch.detach()和 torch.data 的区别是，在求导时，torch.detach()会检查张量的数据是否发生变化，而 torch.data 则不会去检查。新的 tensor 和原来的 tensor 共享数据内存，但不涉及梯度计算，即 requires_grad=False。修改其中一个 tensor 的值，另一个也会改变，因为是共享同一块内存，但如果对其中一个 tensor 执行某些内置操作，则会报错，例如 resize*、resize_as\*、set*、transpose\*。
- 关于 tensor.detach()与 tensor.data:x.data 和 x.detach()新分离出来的 tensor 的 requires_grad=False，即不可求导时两者之间没有区别，但是当 requires_grad=True 的时候的两者之间的是有不同：x.data 不能被 autograd 追踪求微分，但是 x.detach 可以被 autograd()追踪求导。
- with t.no_grad(): 在应用阶段，不需要使用梯度，那么可以使用这个去掉梯度
- 如果在更新的时候不调用 optimizer.zero_grad，两次更新的梯度会叠加。
- 使用 require_grad=False 可以冻结神经网络某一部分的参数，更新的时候就不能减 grad 了
- tensor.item()，直接返回一个数据，但是只能适用于 tensor 里头只有一个元素的情况，否则要是用 tolist()或者 numpy()
- 不建议使用 inplace 操作
- hard replacement 每隔一定的步数才更新全部参数，也就是将估计网络的参数全部替换至目标网络而 soft replacement 每一步就更新，但是只更新一部分(数值上的一部分)参数。比如 theta_new = theta_old _0.95 + theta_new_0.05
- pytorch 官网上有的 Qlearning 例子:<https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html>
- nn.Module.eval()递归调用子模块，将 Module.train 改成 false
- 类似于 tensor.pow, tensor.sum, tensor.mean, tensor.gather 这些操作都可以使用 torch.pow(tensor,\*args)等来代替，使用 t.pow 这种类型的函数可以直接知道它的参数（dim=？之类的），用 tensor.pow 的话可能会因为识别不出来这是个 tensor，导致这个方法出不来。（比如说 a=t.ones((1,1,1)),b=a+a，调用 b.sum 的时候按 TAB 就出不来)
- 同上一条，在传参的时候尽量把参数的类型写清楚，不然在下面使用的时候按 tab 也出不来，十分难顶。例如

  ```python
  def forward(self, x:t.Tensor)->t.Tensor:
      return self.net(x).squeeze(1)
  ```

### 2. 关于 nn.Module.eval()

- net.eval()并不是一种局部禁用梯度计算的机制,不与 requiregrad=False 等价

- 从功能上来说，eval 和 t.no_grad 和 inference 模式是一样的， eval 会影响到模型的训练当且仅当某些模块出现在你的网络中，如 BatchNorm 何 Dropout2d 之类的

- 如果你的网络中出现了 nn.Dropout 或者 nn.Batchnorm2d 这种模块，需要调用 model.eval()和 model.train()，因为它们在两种模式中的表现不一样。

- 不管怎样还是推荐使用 model.train()和 model.eval()，因为你正在使用的模型可能在 eval 和 train 两种模式下表现不同，而你自己不知道。

### 3. 关于GYM 环境

- 经典控制问题，discrete
- Atari Games
- Mujuco

- Copy 和 DeepCopy：是否生成了新的对象？
- Soft Update 的时候，要用 param.data.copy\_不要直接用 param.copy\_，会报错 a leaf Variable that requires grad is being used in an in-place operation.

### 4. 关于 Actor 的输出

- 连续动作

  Actor 输出 mean 和 std，比如说 SAC 里面的，之后根据 mean 和 std 的正态分布进行采样，保持随机性
- 离散动作

  Actor 输出动作的概率，而不是由 mean 和 std 所决定的密度函数，根据概率进行采样，如 AC 里面的，根据概率进行采样来保持随机性
- 梯度传播问题
    1. 对于带权重的参数更新，如 PG，AC，A3C，PPO，使用采样动作的 log_prob 进行梯度回传
    2. 对于要将采样动作放进 Critic 里面计算动作-状态价值的，如 SAC，DDPG，TD3，等，如果他们需要对动作进行采样（尤其是 SAC，采用 action~N(mean,std)进行采样），那么必须使用使用重参数技巧使梯度得以回传，否则直接丢进 critic 就行。

### 5. 关于使用经验池的**AC 架构**算法调参

- 所谓的 AC 架构算法有，DDPG TD3 SAC DQN with PER DQN with HER 等等，他们不是采用带权重的梯度上升，所以是 AC 架构
- 超参数一般有

    ```python
    mem_size
    batch_size
    tau
    gamma
    lr_c
    lr_a
    ```

- 其中对于性能(收敛速度)影响较大的是 tau,lr_c,lr_a,batch_size
- 一定要选好这几个参数，不然网络收敛速度很慢，较差的参数要四五个小时，较好的，半个小时就行

---

## 6. 参考资料

- [莫烦 python](https://mofanpy.com/)

- [《动手学深度学习》](https://zh-v2.d2l.ai/) 这个书里头包含了一些深度学习基础知识和一些热门知识。

- [pytorch 教程](https://www.youtube.com/watch?v=exaWOE8jvy8&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4)

- [OpenAI Gym](https://gym.openai.com/)

- [17 种深度强化学习算法用 Pytorch 实现](https://blog.csdn.net/tMb8Z9Vdm66wH68VX1/article/details/100975138?spm=1001.2101.3001.6650.14&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Edefault-14.no_search_link&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Edefault-14.no_search_link)

- [hhy_csdn 博客-关于强化学习](https://blog.csdn.net/hhy_csdn)

- [PG 算法](https://tomaxent.com/2019/04/14/%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6%E6%96%B9%E6%B3%95/)

- [什么是强化学习](https://paperexplained.cn/articles/article/detail/33/)

- [Markov Chain Monte Carlo Without all the Bullshit](https://jeremykun.com/2015/04/06/markov-chain-monte-carlo-without-all-the-bullshit/)

- [马尔科夫决策与平稳分布](https://blog.csdn.net/qq_34652535/article/details/85343518)

- [深度强化学习基础-王树森](https://www.youtube.com/watch?v=vmkRMvhCW5c&list=PLvOO0btloRnsiqM72G4Uid0UWljikENlU) 这个是讲理论的

- [深入浅出强化学习](https://daiwk.github.io/posts/rl.html)

- [Machine Learning with Phil](https://www.youtube.com/channel/UC58v9cLitc8VaCjrcKyAbrw) 这个是教你跟他敲代码的

- [Maziar Raissi YouTube](https://www.youtube.com/channel/UCxEiGqJ2e-Mg9oQMjVv6poQ)

- [Tianhong Dai 的 GitHub](https://github.com/TianhongDai) 里面有多种强化学习的Pytorch实现

- [DPPO](https://github.com/ZYunfeii/DRL_algorithm_library)

- [OPENAI spinning up](https://spinningup.qiwihui.com/zh_CN/latest/user/introduction.html) OPAI代码库，支持TF Pytorch

- [lilianweng的博客 - Policy Gradient Methods](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/) , 建议有时间可以看看他的主页，里头也有一些东西，比如Transformer，课程学习等等

- [上面博客的中文版](https://tomaxent.com/2019/04/14/%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6%E6%96%B9%E6%B3%95/) 有人翻译过来了

- [DAVID SILVER公开课](https://www.davidsilver.uk/teaching/)

---

## 7. TODO

1. OpenAI spinning up 好好看看
2. ACER的代码重构一下
3. TRPO的代码需要重写
4. 增加课程学习的相关内容
