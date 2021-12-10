# Asynchronous Advantage Actor Critic (A3C)

Deep Mind 提出的一种解决 Actor Critic 不收敛问题的算法，它会创建多个并行的环境, 让多个拥有副结构的 agent 同时在这些并行环境上更新主结构中的参数. 并行中的 agent 们互不干扰, 而主结构的参数更新受到副结构提交更新的不连续性干扰, 所以更新的相关性被降低, 收敛性提高。<br>
在不同线程上使用不同的探索策略，使得经验数据在时间上的相关性很小。这样不需要 DQN 中的 experience replay 也可以起到稳定学习过程的作用，意味着学习过程可以是 on-policy 的。

---

# 与 DDPG 的不同

- A3C 里面有多个 agent 对网络进行异步更新，相关性较低
- 不需要积累经验，占用内存
- on-policy 训练
- 多线程异步
