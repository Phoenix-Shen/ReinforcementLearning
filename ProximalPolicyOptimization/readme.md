# PPO onpolicy 算法

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
