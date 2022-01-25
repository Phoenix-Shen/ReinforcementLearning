# TwinDelayedDeepDeterministicPolicyGradient(TD3)

TD3 的算法没啥问题，但是我写的代码似乎效果很差，不知道是哪个部分出了问题。

---

TD3:DDPG 的改进方法

- Twin：采用类似于 DoubleDQN 的方式解决了 DDPG 中 Critic 对动作的过估计（over estimate）问题
- Delayed：延迟 Actor 更新，更加稳定
- 在 Actor 目标网络输出动作 A’加上噪声，增加算法稳定性。

- ![](aclosspendulum.png)
- ![](Rpendulum.png)

# 参数

1. 激进的 tau（如 0.1，0.5 等）会扰乱训练过程，而过小的 tau（0.005）会使网络收敛很慢，选择一个正确的 tau 对于网络收敛很有用

2. 同 1，学习率也会极大影响速度。

3. batch_size 对训练的影响尚不清楚

4. 要选择合适的 policy_noise、noise_clip，否则会影响到网络对于价值函数的判断，从而导致损失降下去了，但是训练效果不好

# PER 多大程度上会拖慢学习速度？

for Pendulum-v1

with PER

![](./WithPER.png)

NO PER

![](NoPER.png)

因为这两个是同时开始训练的，但是它们的进度不同，说明 PER 还是拖累了训练速度，无 PER 比有 PER**快了 17%**

# HER

搞得不是很清楚
**只适用于 pendulum 环境**

_因为它的 goal 计算和 reward 计算是因环境而异的，没有统一的解决方案_

1. 解决的问题：在稀疏奖励的环境中训练 agent
2. 假设条件：知道要学习的目标 goal，知道如何优化奖励函数 reward function
3. 主要想法：将 goal 加入 transition tuples(s,a,r,s',goal)利用 goal 之间的相似性，使得完成不同 goal 的 transition tuples 可以辅助其它的 goal 做训练

局限：

1. 目标 goal 有时候不清楚
2. 不知道如何优化奖励函数 reward function
3. s 不能被修改的情况
