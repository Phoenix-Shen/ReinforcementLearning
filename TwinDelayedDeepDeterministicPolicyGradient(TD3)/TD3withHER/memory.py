# %%
import gym
import torch as t
import numpy as np


class Replay_buffer():
    def __init__(self, buffer_size, batch_size, n_features, n_actions) -> None:
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n_features = n_features
        self.n_actions = n_actions
        self.memory = t.zeros((buffer_size, n_features*2+2+n_actions))
        self.memory_ptr = 0

    def store_transition(self, s: np.ndarray, a: np.ndarray, r: float, done: float, s_: np.ndarray):
        s = t.tensor(s, dtype=t.float32)
        a = t.tensor(a, dtype=t.float32)
        r = t.tensor(r, dtype=t.float32).unsqueeze(-1)
        done = t.tensor(done, dtype=t.float32).unsqueeze(-1)
        s_ = t.tensor(s_, dtype=t.float32)

        transition = t.hstack(([s, a, r, done, s_]))

        index = self.memory_ptr % self.buffer_size
        self.memory[index, :] = transition
        self.memory_ptr += 1

    def sample(self):
        index = np.random.choice(
            min(self.memory_ptr, self.buffer_size), size=self.batch_size)

        states = self.memory[index, :self.n_features]
        actions = self.memory[index,
                              self.n_features:self.n_features+self.n_actions]
        rewards = self.memory[index, self.n_features +
                              self.n_actions:self.n_features+self.n_actions+1]
        dones = self.memory[index, self.n_features +
                            self.n_actions+1:self.n_features+self.n_actions+2]
        states_ = self.memory[index, self.n_features+self.n_actions+2:]

        return states, actions, rewards, dones, states_


"""#%%
import numpy as np
mem = Replay_buffer(1,1,8,2)
s = np.ones((8))
s_ = np.ones((8))*2
a = np.zeros((2))+5
r = 114514
done = False
mem.store_transition(s,a,r,done,s_)
# %%
"""


class SumTree():
    def __init__(self, max_mem) -> None:
        self.capacity = max_mem
        self.data_ptr = 0
        # 对于capacity个数据节点，我们需要使用capacity-1个父节点来形成一个树形结构
        # 那么一共需要 2*capacity-1个节点，定义tree[:self.capacity-1]为父节点，后面的是叶子节点
        self.tree = np.zeros(2*self.capacity-1)
        # 存储数据的节点，与tree[self.capacity-1:] 一一对应
        self.data = np.zeros(self.capacity, dtype=object)

    def add(self, priority: float, data: object):
        # 叶子节点的下标
        tree_idx = self.data_ptr+self.capacity-1
        # 存储数据
        self.data[self.data_ptr] = data
        # 更新父节点的权重
        self.update_parent_weights(tree_idx, priority)
        # 如果达到了存储的上限，则踢掉刚刚进来的数据
        self.data_ptr += 1
        if self.data_ptr >= self.capacity:
            self.data_ptr = 0

    def update_parent_weights(self, tree_idx: int, priority: float):
        # 获得修改的值
        change_value = priority - self.tree[tree_idx]
        # 直接修改叶子节点的值
        self.tree[tree_idx] = priority
        # 跟这个叶子节点有关的父亲节点 全部要更新
        while tree_idx != 0:
            tree_idx = (tree_idx-1)//2
            self.tree[tree_idx] += change_value

    def get_leaf(self, v: float):
        """
        根据采样到的值v来取叶子节点的数据
        """
        parent_idx = 0
        while True:
            cl_idx = 2*parent_idx+1
            cr_idx = cl_idx+1
            # 超出界限了就结束搜索
            if cl_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v = v-self.tree[cl_idx]
                    parent_idx = cr_idx
        data_idx = leaf_idx-(self.capacity-1)
        # return leaf_idx,and the priority of the idx, and the data
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self):
        return self.tree[0]


class ProritizedExperienceReplay():
    def __init__(self,
                 max_size: int,
                 batch_size: int,
                 clipped_abs_error: float,
                 epsilon: float,
                 alpha: float,
                 beta: float,
                 beta_increment: float,
                 ) -> None:
        """
        Params
        ------
        alpha:trade-off factor 控制采样在uniform和greedy的偏好,0代表均匀随机，1代表完全按贪婪算法
        epsilon: 避免除数为0的一个很小的数
        beta: 决定抵消PER对于收敛结果的影响，1代表完全抵消，这样就与ER没有区别了
        beta_increment: 每次采样对beta采取的增量
        clipped_abs_error: abs_error的上限
        """

        # 存储参数
        self.clipped_abs_error = clipped_abs_error
        self.batch_size = batch_size
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.memory_ptr = 0
        # 定义SUMTREE
        self.tree = SumTree(max_size)
    #states, actions, rewards, dones, states_

    def store_transition(self, state, action, reward,  done, state_):
        transition = (np.expand_dims(state, axis=0), np.expand_dims(
            action, axis=0), reward, done, np.expand_dims(state_, axis=0))
        # 索引后面的叶子节点
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = self.clipped_abs_error
            # 存进来的时候，调成最大的优先级，保证每个样本都要过一遍
        self.tree.add(max_priority, transition)
        self.memory_ptr += 1

    def sample(self) -> tuple[np.ndarray, t.Tensor, t.Tensor, t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
        """
        Returns
        -------
        Tree index : the indexs of the tree
        Data : the data of the nodes
        ISWeights : the normalized Priority of the nodes
        """
        b_idx = np.empty((self.batch_size), dtype=np.int32)

        ISWeights = np.empty((self.batch_size, 1))
        # 分成batch_size 个段
        priority_segments = self.tree.total_priority/self.batch_size
        # 递增的beta
        self.beta = np.min([1., self.beta+self.beta_increment])
        # 取得最小倍率，这是normalization的过程
        min_prob = np.min(
            self.tree.tree[-self.tree.capacity])/self.tree.total_priority
        samples = []
        for i in range(self.batch_size):
            # 将优先值分为batch_size片，然后从每片中均匀采样
            a, b = priority_segments*i, priority_segments*(i+1)
            sample_val = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(sample_val)
            # 取得优先值的占比
            prob = p/self.tree.total_priority
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i] = idx
            samples.append(data)
        state, action, reward, done, state_, = zip(*samples)
        # 返回抽样值
        return b_idx, t.tensor(np.concatenate(state, axis=0)), t.tensor(np.concatenate(action, axis=0)), t.tensor(np.array(reward)).unsqueeze(-1), t.tensor(np.array(done)).unsqueeze(-1),  t.tensor(np.concatenate(state_, axis=0)),  t.tensor(ISWeights)

    def batch_update(self, tree_idx: np.ndarray, abs_errors: np.ndarray):
        # + epsilon避免0的阿尔法次方
        abs_errors += self.epsilon
        clipped_errors = np.minimum(abs_errors, self.clipped_abs_error)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update_parent_weights(ti, p)
