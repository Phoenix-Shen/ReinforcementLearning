# 关于 Qlearning收敛性的证明

## 贝尔曼最优方程

对于马尔科夫决策过程来说，我们的状态价值函数state-value function 和 动作状态价值函数 action-state value funtion分别满足Bellman方程：

$$
\begin{aligned}
V_\pi(s)  &= \sum_{a \in \mathcal{A}} \pi(a\vert s) \left (R_s^a + \gamma \sum_{s^\prime \in \mathcal{S} }\mathcal{P}_{s s^\prime}^a V_\pi (s^\prime)\right)\\

q_\pi(s,a) &= R_s^a + \gamma \sum_{s^\prime \in \mathcal{S}}\mathcal{P}_{s s^\prime} ^a\sum_{a^\prime \in \mathcal{A}} \pi(a^\prime \vert s^\prime) q_\pi(s^\prime,a^\prime)
\end{aligned}
$$

如果我们已知reward function $R_s^a$和状态转移函数$\mathcal{P}_{s s^\prime} ^a$那么就可以通过解Bellman方程来获得策略$\pi$下面的动作值函数，但这通常会面临计算量较大的问题。

## 最优值函数

我们假设有一个最优的值函数，他是所有策略下面对值函数最好的估计：

$$V^* (s)  = \underset{\pi}{\max} \ V_\pi(s)$$

同样地，我们有：

$$Q^*(s,a) = \underset{\pi}{\max} \ Q_\pi(s,a)$$

我们把能够实现以上最优值函数的策略叫做$\pi^*$，它满足以下定理：

- 对于任意的MDP，总有一个$\pi^*$，它比其它策略更好或者相等
- 所有的最优策略会让状态值函数取得最高
- 所有的最优策略会让动作-状态值函数取得最高

值函数包含动作状态函数就是对状态或者状态加动作的收益估计，让收益最大化的策略就是最好的策略

于是我们可以写下关于$V^*(s),Q^*(s,a)$的Bellman最优方程：
$$
\begin{aligned}
V^*(s) = \underset{a}{\max} \left(R_s^a + \gamma \sum_{s^\prime \in \mathcal{S} }\mathcal{P}_{s s^\prime}^a V^*(s^\prime)\right)\\

Q^*(s,a) = R_s^a + \gamma \sum_{s^\prime \in \mathcal{S} }\mathcal{P}_{s s^\prime} ^a \underset{a^\prime}{\max} \ Q^*(s^\prime,a^\prime)

\end{aligned}

$$

我们选取动作就是需要让Q值最大：
$$
\pi^*(a\vert s ) =
\begin{cases}
  1  \ if \ a= \underset{a\in \mathcal{A}}{\argmax} \ Q^*(s,a) \\
  0 \  otherwise
\end{cases}
$$

## 值迭代

我们尝试使用value iteration 的方法来解Bellman方程，这种方法属于动态规划，它将MDP问题分成两个子问题：

- 一个当前情况$s$下的最优动作
- 在后续状态$s ^\prime$下沿着最优策略执行

最优性定理告诉我们：
一个策略$\pi(s\vert s)$在$s$上去的最优值函数$V_\pi(s) = V_\pi^*(s)$当且仅当：对于从状态$s$可以到达的任何状态$s^\prime$，$\pi$从$s^\prime$中能够获得最优值函数$V_\pi(s^\prime) = V_\pi^*(s^\prime)$

因为我们有：
$$V^*(s) = \underset{a}{\max} \left(R_s^a + \gamma \sum_{s^\prime \in \mathcal{S} }\mathcal{P}_{s s^\prime}^a V^*(s^\prime)\right)$$

而且最终状态的值函数已经被确定，所以我们一直倒着推，就可以确定所有状态$s$的最优状态值函数。

## 压缩映射定理

需要证明上述的值迭代过程会收敛到$V^*$。

根据压缩映射定理，我们有：

对于任何在算子$T(v)$下完备（即封闭）的度量空间$V$，如果算子$T$为$\gamma$压缩，则会有：

- $T(v)$会收敛到一个固定点$v^*$
- 收敛速度线性正比于$\gamma$

我们将$ T(v) =\underset{a}{\max} \left(R_s^a + \gamma \sum_{s^\prime \in \mathcal{S} }\mathcal{P}_{s s^\prime}^a V^*(s^\prime)\right)$看做算子，称为Bellman optimality backup operator，需要证明它为$\gamma$压缩。

如果一个算子$T$为$\gamma$压缩，那么有：

$${\vert \vert T(u) - T(v) \vert \vert }_{\infty} \leq \gamma {\vert \vert u - v \vert \vert }_{\infty}, \gamma \lt 1$$

其中 ${\vert \vert u - v \vert \vert }_{\infty} = \underset{s \in \mathcal{S}}{\max}\vert u(s) -v(s) \vert$ 表示两个值函数在任意状态下的最大差距。

展开${\vert \vert u - v \vert \vert }_{\infty}$：

$${\left \vert \left\vert {\underset{a}{\max} \left(R_s^a + \gamma \sum_{u \in \mathcal{S} }\mathcal{P}_{s s^\prime}^u V^*(u)\right) - \underset{a}{\max} \left(R_s^a + \gamma \sum_{v \in \mathcal{S} }\mathcal{P}_{s s^\prime}^v V^*(v)\right) } \right \vert \right \vert }_{\infty}\\
= {\vert \vert \gamma P^a u -    \gamma P^a v\vert \vert}_{\infty} \\
\le \gamma {\vert \vert u - v\vert \vert}_{\infty}$$

因为Bellman Optimal Equation：$ V^*= T(V^*)$
,所以它会收敛到最优值函数。

## 从值迭代到Q-learning

之迭代是在知道环境的情况下，Q-learning是不知道情况的环境下，我们可以通过采样来代替环境，两者本质上没有什么区别。
