# Asynchronous Advantage Actor Critic (A3C)

Deep Mind 提出的一种解决 Actor Critic 不收敛问题的算法，它会创建多个并行的环境, 让多个拥有副结构的 agent 同时在这些并行环境上更新主结构中的参数. 并行中的 agent 们互不干扰, 而主结构的参数更新受到副结构提交更新的不连续性干扰, 所以更新的相关性被降低, 收敛性提高。<br>
在不同线程上使用不同的探索策略，使得经验数据在时间上的相关性很小。这样不需要 DQN 中的 experience replay 也可以起到稳定学习过程的作用，意味着学习过程可以是 on-policy 的。

---

# 与 DDPG 的不同

- A3C 里面有多个 agent 对网络进行异步更新，相关性较低
- 不需要积累经验，占用内存
- on-policy 训练
- 多线程异步

---

## python 多线程无法占用：使用 torch.multiprocessing 的包

- Python 由于全局锁 GIL 的存在，无法享受多线程带来的性能提升。

- multiprocessing 包采用子进程的技术避开了 GIL，使用 multiprocessing 可以进行多进程编程提高程序效率。
- multiprocessing 使用`共享内存`进行进程中的通信
- 模型并行：把模型拆分放到不同的设备进行训练
- 数据并行：把数据切分，并复制到各个机器上，然后将所有结果按照某种算法 hebing
- https://ptorch.com/news/176.html
