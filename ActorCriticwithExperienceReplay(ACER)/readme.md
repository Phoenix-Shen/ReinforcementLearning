# ACER

`Actor-Critic的off-policy算法`

# 它不是有效的，它不能够学到东西，为什么不能够使用 ER 呢？

- policy gradient 估计的是经验的梯度，如果使用旧的 policy 产生的样本，那么我们就不在估计当前 policy 的梯度了，所以 policy gradient 不能使用 experience，就相当于你使用几十年前的房价来对比现在的房价一样，更新策略之后，之前的行为就不再是我这个策略做出来的了，所以我们不能够使用 experience

- Qlearning 学习的是 Q 值，它与上下状态是没有关系的，但是 AC 是学习一个策略值，这个值需要下一个状态的采样，也就是说它与下一个状态是相关的，而 Qlearning 只是需要一个最大化的操作

- 我们需要使用 importance sampling 来使 policy gradient 或者 acer 变成 off policy

- off-policy：使值最大（即获得最多的 reward） on-policy：根据 π_theta 来选择当前的动作
