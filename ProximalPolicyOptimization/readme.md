## PPO / TRPO

TRPO 将 θ 和 θ'的分布写在了约束里面
PPO 写在了 loss 损失里面 βKL(θ，θk)

# importance Sampling

这个理论支持了我们从 ON-policy 转向 OFF-policy

# ON-policy 和 OFF-policy

on：与环境交互的这个 agent 就是我们要学习的 agent，off：不一定是这个 agent。
