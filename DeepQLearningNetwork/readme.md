QLearning 的局限性：Q 表无法进行所有情况的枚举，在某些情况下是不可行的，比如下围棋。

使用神经网络进行该操作就可以解决该问题。

f(state,action)->action.value
或者是
f(state)->list<actionvalue>

Features: Expericence Replay and Fixed Q-targets

Experience Replay : 将每一次实验得到的惊艳片段记录下来，然后作为经验，投入到经验池中，每次训练的时候随机取出一个 BATCH，可以复用数据。

Fixed Q-target: 在神经网络中，Q 的值并不是互相独立的，所以不能够进行分别更新操作，那么我们需要将网络参数复制一份，解决该问题。

为了解决 overestimate 的问题，引入 double DQN，算法上有一点点的改进
