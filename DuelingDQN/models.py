import torch as t
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import time
################
# DATA STRCTURE#
################


class SumTree(object):
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2*capacity-1)
        # 存储capacity个数据，一共需要capacity-1个父节点
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
    # 加入节点操作

    def add(self, p, data):
        tree_idx = self.data_pointer+self.capacity-1  # 下一个数据存储在树的哪里呢？
        self.data[self.data_pointer] = data  # 更新数据节点

        self.update(tree_idx, p)
        self.data_pointer = self.data_pointer+1
        # 如果到达了上限的话，就替换掉原来的数据
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
    # 更新节点的优先值，不使用递归，递归导致速度慢
    # 更新下标为tree_idx的节点的所有值

    def update(self, tree_idx: int, p):
        change = p-self.tree[tree_idx]  # 获得改变的值
        self.tree[tree_idx] = p
        # 将这个改变传递给所有父节点
        # 当tree_idx指向父节点的时候才停下来
        while tree_idx != 0:
            tree_idx = (tree_idx-1)//2
            self.tree[tree_idx] += change

    # 根据vlaue选取一个叶子节点,采用while循环增加速度
    def get_leaf(self, v):
        parent_idx = 0
        while True:
            # 取得父节点的子树下面的左右节点
            cl_idx = 2*parent_idx+1
            cr_idx = cl_idx+1
            if cl_idx >= len(self.tree):
                # 如果大于树的长度，那么这个就是我们要找的
                leaf_idx = parent_idx
                break
            # 小于等于该节点的值，则继续向下查找，否则减去这个节点的值，再继续向下查找
            else:
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx
        data_idx = leaf_idx-self.capacity+1  # 获取当前在数据中的下标
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]
    # 获取总的有优先值

    @property
    def total_p(self):
        return self.tree[0]


# 存储每个步骤（经验） (s,a,r,s_)
class MEMORY_BUFFER_PER(object):
    epsilon = 0.01
    alpha = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # 损失的最大值

    def __init__(self, capacity: int) -> None:
        super().__init__()
        self.Sumtree = SumTree(capacity=capacity)

    # 存储一系列动作(s,a,r,s_)在一开始的时候，所有节点的优先值是self.abs_err_upper
    def store(self, transition):
        max_p = np.max(self.Sumtree.tree[-self.Sumtree.capacity])  # 取得最大的优先值
        if max_p == 0:
            max_p = self.abs_err_upper
        self.Sumtree.add(max_p, transition)

    # 获得batch_size的样本,ISWeights=importance sampling weights
    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n), dtype=np.int32), np.empty(
            (n, self.Sumtree.data[0].size)), np.empty((n, 1))
        # 进行分片
        pri_seg = self.Sumtree.total_p/n
        self.beta = np.min(
            [1., self.beta+self.beta_increment_per_sampling])  # beta的最大值是1
        # 之后计算ISWeight要用
        min_prob = np.min(
            self.Sumtree.tree[-self.Sumtree.capacity:])/self.Sumtree.total_p
        # 进行采样
        for i in range(n):
            a, b = pri_seg*i, pri_seg*(i+1)
            v = np.random.uniform(a, b)
            # 根据随机获得的V值进行一个数据的获取
            idx, p, data = self.Sumtree.get_leaf(v)
            prob = p/self.Sumtree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    # 根据误差来更新一个batch
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # 防止除以0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.Sumtree.update(ti, p)

############
# DQN MODEL#
############


class _DQN(nn.Module):
    def __init__(self, n_states: int, n_actions: int, hidden_layers: int):
        super().__init__()
        self.input_size = n_states
        self.output_size = n_actions
        self.hidden_layers = hidden_layers

        self.feature_layer = nn.Linear(self.input_size, self.hidden_layers)
        self.value_layer = nn.Linear(self.hidden_layers, 1)  # 输出value
        self.advantage_layer = nn.Linear(
            self.hidden_layers, self.output_size)  # 输出每个动作的优先值

    def forward(self, x: t.Tensor) -> t.Tensor:
        feature1 = F.relu(self.feature_layer(x))
        feature2 = F.relu(self.feature_layer(x))
        # value
        value = self.value_layer(feature1)
        # advantage
        advantage = self.advantage_layer(feature2)
        # 其实在这里写advantage.mean(dim=1).expand_as(advantage)也是可以的
        # advantage满足 sum(advantage)=0
        return value+advantage-advantage.mean(dim=1, keepdim=True)

#########################
#DOUBLE DQN ARCHETECTURE#
#########################


class DuelingDQN(nn.Module):
    def __init__(self,
                 n_states: int,
                 n_actions: int,
                 hidden_layers: int,
                 lr=0.001,
                 memory_size=100000,
                 target_replace_iter=100,
                 batch_size=32,
                 reward_decay=0.9,
                 e_greedy=0.9) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.n_states = n_states
        self.train_net = _DQN(n_states, n_actions, hidden_layers)
        self.target_net = _DQN(n_states, n_actions, hidden_layers)
        # 复制参数
        self.target_net.load_state_dict(self.train_net.state_dict())
        self.optimizer = optim.RMSprop(
            self.train_net.parameters(), lr=lr, eps=0.001, alpha=0.95)
        self.memory = MEMORY_BUFFER_PER(memory_size)
        self.learn_step_counter = 0
        self.epsilon = e_greedy
        self.target_replace_iter = target_replace_iter
        self.batch_size = batch_size
        self.gamma = reward_decay

    def choose_action(self, x):
        # state传进来转tensor扩展维度
        x = t.FloatTensor(x).unsqueeze(dim=0)
        # 有一定的随机性
        if np.random.uniform() < self.epsilon:
            actions_value = self.target_net(x)
            action = t.argmax(actions_value).data.numpy()
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def store_transition(self, s, a, r, s_):
        # np.hstack将参数元组的元素数组按水平方向进行叠加
        transition = np.hstack((s, [a, r], s_))
        self.memory.store(transition)

    def learn(self):
        # target parameter update
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.train_net.state_dict())
        self.learn_step_counter += 1

        tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
        # 转换成tensor
        b_s = t.FloatTensor(batch_memory[:, :self.n_states])
        b_a = t.LongTensor(
            batch_memory[:, self.n_states:self.n_states+1].astype(int))
        b_r = t.FloatTensor(batch_memory[:, self.n_states+1:self.n_states+2])
        b_s_ = t.FloatTensor(batch_memory[:, -self.n_states:])
        # 通过神经网络得出q的值
        q_eval = self.train_net(b_s).gather(1, b_a)
        # 慎用detach，难顶
        q_next = self.target_net(b_s_).detach()

        ######################################
        #Tensor.max(dim:int)                 #
        #a=t.FloatTensor([[1],[2],[3],[4]])  #
        #torch.return_types.max(             #
        #values=tensor([1., 2., 3., 4.]),    #
        #indices=tensor([0, 0, 0, 0]))       #
        ######################################

        q_target = b_r+self.gamma*q_next.max(1)[0].unsqueeze(1)
        # 其实这三步看下来就是MSE
        loss = (q_eval-q_target).pow(2)*t.FloatTensor(ISWeights)
        prios = loss.data.numpy()
        loss = loss.mean()
        self.memory.batch_update(tree_idx, prios)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self):
        t.save(self.train_net.state_dict(),
               "{}-{}-{} {}-{}-{}.pth".format(time.localtime()[0],
                                              time.localtime()[1],
                                              time.localtime()[2],
                                              time.localtime()[3],
                                              time.localtime()[4],
                                              time.localtime()[5], ))

    def load(self, path: str):
        model = t.load(path)
        self.train_net.load_state_dict(model)
        self.target_net.load_state_dict(model)
