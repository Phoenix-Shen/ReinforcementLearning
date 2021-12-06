QLearning 的局限性：Q 表无法进行所有情况的枚举，在某些情况下是不可行的，比如下围棋。

使用神经网络进行该操作就可以解决该问题。

f(state,action)->action.value
或者是
f(state)->list<actionvalue>

Features: Expericence Replay and Fixed Q-targets

Experience Replay : 将每一次实验得到的惊艳片段记录下来，然后作为经验，投入到经验池中，每次训练的时候随机取出一个 BATCH，可以复用数据。

Fixed Q-target: 在神经网络中，Q 的值并不是互相独立的，所以不能够进行分别更新操作，那么我们需要将网络参数复制一份，解决该问题。

为了解决 overestimate 的问题，引入 double DQN，算法上有一点点的改进

```
Deep Q Network
融合Qlearning和神经网络

1、传统方法劣势
    状态过多的时候无法完全枚举（围棋）

2、神经网络如何处理该问题？
    Q值=网络（tuple（状态，动作））
    动作=网络（状态）
    两种方法都省去了Q表的构建操作

3、输入输出
网络输入->状态
输出->每个动作的权重 ，根据最大的权重选择下一步的动作

4、更新策略
q_eval = self.eval_net(b_s).gather(1, b_a)

q_next = self.target_net(b_s_).detach()

q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)

loss = self.loss_func(q_eval, q_target)

MSE（估计值，实际值），再进行反向传播


```
