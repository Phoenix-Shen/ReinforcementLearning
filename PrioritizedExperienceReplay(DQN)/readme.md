# DQN with Prioritized Replay

与传统的 DQN 算法的结构类似，但是它把已存储的步骤按照重要性进行了排序，在学习的时候按照优先级来进行抽样学习。

## 具体特征

- 采用按照优先级排列的 Memory
- 按照 TD-error 来决定优先级
- 使用 SumTree 这种数据结构来存储 Memory，从而减少计算

`SumTree可以在model.py里面看到`
