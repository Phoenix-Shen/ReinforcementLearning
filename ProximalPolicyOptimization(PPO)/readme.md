# PPO onpolicy 算法

- 主要为了解决 actor critic 训练步长的问题，在这里实现了 ppo-clip
- On-policy 是要用 πθ 收集数据，当 θ 更新了，我们就要重新进行数据收集。
- 这个方法十分慢，我们能不能使用 πθ'收集数据，把这个数据给 πθ 使用进行训练，由于 θ'是不变的，那我们就可以进行数据重用。
- gradient for update： advantage\* gradientprobablity ,advantage 表示的是从这一步能够获得多大的益处

## importance Sampling

- 这个理论支持了我们从 ON-policy 转向 OFF-policy
- 问题：当采样不够的时候，这个方法效果就很差了
- 在应用阶段使用 θ 进行数据采样，然后积累经验，进行参数的更新

## PPO / TRPO

TRPO 将 θ 和 θ'的分布写在了约束里面
PPO 写在了 loss 损失里面 βKL(θ，θk)

## ON-policy 和 OFF-policy

on：与环境交互的这个 agent 就是我们要学习的 agent，off：不一定是这个 agent。

## 在代码中，将会实现 PPO-Clip 算法，这是 OpenAI 提出的

效果很差，因为“偶尔的胜利”不足以使网络的参数完全修正，但是 offpolicy 的 dqn with per 能够多次学习成功的经验，所以对于这个 Pendulum-v1 来说，8 太行

# TrustRegionPolicyOptimization

- TRPO 算法 (Trust Region Policy Optimization)和 PPO 算法 (Proximal Policy Optimization)都属于 MM(Minorize-Maximizatio)算法
- MM 算法是一种迭代优化方法，它利用函数的凸性来找到它们的最大值或最小值。
- policy gradient 的缺点：步长（lr）不好控制，而当步长不对的时候，PG 学习到的策略会更差，导致越学越差最后崩溃。
- 当策略更新后，回报函数的值不能更差，我们要找到新的策略使回报函数不减小，这就是 TRPO 解决的问题
- 要想不减，自然想到一个方法：theta=theta+一个正值
- ![](./algorithm.png)
- 目标：最大化优势函数，并且使新旧策略的差异不能够太大
- https://blog.csdn.net/tanjia6999/article/details/99716133
