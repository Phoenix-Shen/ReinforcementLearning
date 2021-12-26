# Model Free 算法

1. TRPO、PPO
2. DDPG、D4PG、TD3
3. Soft Q-Learning、Soft Actor-Critic

# Soft Actor Critic

面向 Maximum Entropy Reinforcement Learning 开发的一种 off-policy 算法，使用随机策略  
对比 DDPG，它不遗漏每一个好的动作

# 基于最大熵的 RL 算法

1. 可以学到所有的最优路径
2. 更强的探索能力
3. 更强的鲁棒性（面对干扰的时候能够自我调整）

# 三个关键技术

1. 采用分离策略网络以及值函数网络的 AC 架构
2. ER 能够使用历史数据，高效采样
3. 熵最大化以鼓励探索

# SAC 要学习的东西

1. Policy PI_theta
2. Soft Q value function Q_omega
3. Soft State-value function V_psi

![](./SAC_algo.png)

soft 到底是什么 不太了解
