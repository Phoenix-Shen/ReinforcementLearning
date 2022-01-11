# TwinDelayedDeepDeterministicPolicyGradient(TD3)

DDPG 的改进方法

- Twin：采用类似于 DoubleDQN 的方式解决了 DDPG 中 Critic 对动作的过估计（over estimate）问题
- Delayed：延迟 Actor 更新，更加稳定
- 在 Actor 目标网络输出动作 A’加上噪声，增加算法稳定性。
