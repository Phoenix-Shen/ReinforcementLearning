import torch as t
import torch.nn as nn
import torch.optim as optim
import numpy as np


class SumTree(object):
    def __init__(self, capacity: int) -> None:
        super().__init__()
        self.data_pointer = 0
        self.capacity = capacity
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.tree = np.zeros(2*capacity-1)
        # [--------------data frame-------------]
        #             size: capacity
        self.data = np.zeros(capacity, dtype=object)

    # p means the priority of the data
    # data means the transitions
    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # data frame updation
        self.update(tree_idx, p)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0  # 如果达到上限了，就恢复初始值

    def update(self, tree_idx, p):
        change = p-self.tree[tree_idx]
        self.tree[tree_idx] = p
        # tne propagate the change through the tree
        while tree_idx != 0:
            tree_idx = (tree_idx-1)//2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property  # 这个就跟vue里面的计算属性一样？？
    def totoal_p(self):
        return self.tree[0]  # the root


# stored as ( s, a, r, s_ ) in SumTree
class Memory(object):
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity: int) -> None:
        super().__init__()
        self.tree = SumTree(capacity=capacity)

    # transition is an object
    def store(self, transition: object):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])  # 索引后面的叶子节点
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty(
            (n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.totoal_p/n  # 进行分片
        self.beta = np.min(
            [1., self.beta+self.beta_increment_per_sampling])  # 增加β
        min_prob = np.min(
            self.tree.tree[-self.tree.capacity])/self.tree.totoal_p  # 取得最小的占比

        for i in range(n):
            # 将总的优先值分为n片，然后进行随机采样
            a, b = pri_seg*i, pri_seg*(i+1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p/self.tree.totoal_p  # 获得该优先值的占比
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # 避免除以0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)  # 进行最大值的限制
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)  # 更新这个batch的p值，根据abs_errors


class DQNwithPER(nn.Module):
    def __init__(self,
                 n_features,
                 n_actions,
                 lr=0.001,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=500,
                 memory_size=10000, batch_size=32,
                 e_greedy_increment=None, prioritized=True) -> None:
        super().__init__()
        # hyper parameters
        self.n_features = n_features
        self.n_actions = n_actions
        self.gamma = reward_decay
        self.lr = lr
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.e_greedy_increment = e_greedy_increment
        self.prioritized = prioritized
        self.learn_step_counter = 0
        self.epsilon_max = e_greedy
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.cost_his = []
        # net structure
        self.eval_net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.Tanh(),
            nn.Linear(128, n_actions),
            nn.Softmax(),
        )
        self.target_net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.Tanh(),
            nn.Linear(128, n_actions),
            nn.Softmax(),
        )
        # Memory definition
        if self.prioritized:
            self.memory = Memory(capacity=memory_size)
        else:
            self.memory = np.zeros((self.memory_size, n_features*2+2))
        # optimizer
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=self.lr)

    def choose_action(self, x):
        # 将x转换成32-bit floating point形式，并在dim=0增加维数为1的维度
        x = t.unsqueeze(t.FloatTensor(x), 0)
        # 生成一个在[0, 1)内的随机数，如果小于EPSILON，选择最优动作
        if np.random.uniform() < self.epsilon:
            # 通过对评估网络输入状态x，前向传播获得动作值
            actions_value = self.eval_net.forward(x)
            # 输出每一行最大值的索引，并转化为numpy ndarray形式
            action = t.max(actions_value, 1)[1].data.numpy()
            # 输出action的第一个数
            action = action[0]
        else:                                                                   # 随机选择动作
            # 这里action随机等于0或1 (N_ACTIONS = 2)
            action = np.random.randint(0, self.n_actions)
        # 返回选择的动作 (0或1)
        return action

    def store_transition(self, s, a, r, s_):
        # 在水平方向上拼接数组
        transition = np.hstack((s, [a, r], s_))
        if self.prioritized:
            self.memory.store(transition)
        else:
            if not hasattr(self, "memory_counter"):
                self.memory_counter = 0
            # 如果记忆库满了，便覆盖旧的数据
            # 获取transition要置入的行数
            index = self.memory_counter % self.memory_size
            # 置入transition
            self.memory[index, :] = transition
            # memory_counter自加1
            self.memory_counter += 1

    def learn(self):
        # 一开始触发，然后每100步触发
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(
                self.eval_net.state_dict())         # 将评估网络的参数赋给目标网络

        if self.prioritized:
            tree_idx, b_memory, ISWeights = self.memory.sample(
                self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_size, self.batch_size)
            # 抽取32个索引对应的32个transition，存入b_memory
            b_memory = self.memory[sample_index, :]

        b_s = t.FloatTensor(b_memory[:, :self.n_features])
        # 将32个s抽出，转为32-bit floating point形式，并存储到b_s中，b_s为32行4列
        b_a = t.LongTensor(
            b_memory[:, self.n_features:self.n_features+1].astype(int))
        # 将32个a抽出，转为64-bit integer (signed)形式，并存储到b_a中 (之所以为LongTensor类型，是为了方便后面torch.gather的使用)，b_a为32行1列
        b_r = t.FloatTensor(b_memory[:, self.n_features+1:self.n_features+2])
        # 将32个r抽出，转为32-bit floating point形式，并存储到b_s中，b_r为32行1列
        b_s_ = t.FloatTensor(b_memory[:, -self.n_features:])
        # 将32个s_抽出，转为32-bit floating point形式，并存储到b_s中，b_s_为32行4列

        # 获取32个transition的评估值和目标值，并利用损失函数和优化器进行评估网络参数更新
        q_eval = self.eval_net(b_s).gather(1, b_a)
        # eval_net(b_s)通过评估网络输出32行每个b_s对应的一系列动作值，然后.gather(1, b_a)代表对每行对应索引b_a的Q值提取进行聚合
        q_next = self.target_net(b_s_).detach()
        # q_next不进行反向传递误差，所以detach；q_next表示通过目标网络输出32行每个b_s_对应的一系列动作值
        q_target = q_eval
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = b_memory[:, self.n_features].astype(int)
        reward = b_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = t.FloatTensor(reward) + \
            self.gamma * t.max(q_next)

        if self.prioritized:
            q_eval_update = self.eval_net(
                t.FloatTensor(b_memory[:, :self.n_features]))
            q_next_update = self.target_net(
                t.FloatTensor(b_memory[:, :self.n_features]))
            abs_error = t.sum(t.abs(q_eval_update, q_next_update), dim=1)
            loss_func = t.nn.MSELoss()
            loss = loss_func(q_eval_update, q_next_update)
            self.memory.batch_update(tree_idx, abs_error)
            self.cost_his.append(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.learn_step_counter += 1
