import torch as t
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import time
# actor 负责根据当前状态选择一个Q值最大的动作


class Actor(nn.Module):
    def __init__(self,
                 n_features: int,
                 n_actions: int,
                 boundary: list = [-2, 2]):
        super().__init__()
        # PARAMETERS
        self.n_features = n_features
        self.n_actions = n_actions
        self.boundary = boundary
        # NET STRUCTURE
        self.net = nn.Sequential(
            nn.Linear(self.n_features, 256),
            nn.ReLU(),
            nn.Linear(256, self.n_actions),
            nn.Tanh(),
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        x_ = self.net(x)
        low = self.boundary[0]
        high = self.boundary[1]
        # 归一化到我们想要的区间,从tanh出来的那一层它的范围是[-1,1]
        return low+(high-low)/(1-(-1))*(x_-(-1))


# critic 接受当前状态s和当前动作a作为参数，返回当前动作的Q值
class Critic(nn.Module):
    def __init__(self,
                 n_features: int,
                 n_actions: int,
                 ):
        super().__init__()
        # PARAMETERS
        self.n_features = n_features
        self.n_actions = n_actions

        # NET STRUCTURE
        self.fcs = nn.Linear(self.n_features, 256)
        self.fca = nn.Linear(self.n_actions, 256)
        self.out = nn.Linear(256, 1)

    def forward(self, s: t.Tensor, a: t.Tensor) -> t.Tensor:
        s_v = self.fcs(s)
        a_v = self.fca(a)
        actions_value = self.out(F.relu(a_v+s_v))
        return actions_value


class DDPG(nn.Module):
    def __init__(self,
                 n_features: int,
                 n_actions: int,
                 memory_size: int,
                 replaycement_epoch: int,
                 lr_c=0.002,
                 lr_a=0.001,
                 batch_size=32,
                 reward_decay=0.9,
                 boundary: list = [-2, 2]) -> None:
        super().__init__()

        # PARAMETERS
        self.n_features = n_features
        self.n_actions = n_actions
        self.memory_size = memory_size
        self.replaycement_epoch = replaycement_epoch
        self.lr_c = lr_c
        self.lr_a = lr_a
        self.batch_size = batch_size
        self.reward_decay = reward_decay
        self.boundary = boundary
        self.t_replace_counter = 0
        # ACTOR AND CRITIC
        self.actor = Actor(self.n_features, self.n_actions, self.boundary)
        self.actor_target = Actor(
            self.n_features, self.n_actions, self.boundary)

        self.critic = Critic(self.n_features, self.n_actions)
        self.critic_target = Critic(
            self.n_features, self.n_actions)

        # OPTIMIZER
        self.optimizer_actor = optim.Adam(
            self.actor.parameters(), lr=self.lr_a)
        self.optimizer_critic = optim.Adam(
            self.critic.parameters(), lr=self.lr_c)
        # LOSS FUNCTION
        self.loss_func = nn.MSELoss()
        # MEMORY
        self.memory = np.zeros(
            (memory_size, self.n_features*2+1+self.n_actions))
        self.memory_pointer = 0

    def sample_from_memory(self):
        index = np.random.choice(self.memory_size, size=self.batch_size)
        return self.memory[index, :]

    def store_transition(self, s: list, a, r: float, s_: list):
        # 在水平方向上拼接数组
        transition = np.hstack((s, [a, r], s_))
        # 如果记忆库满了，便覆盖旧的数据
        # 获取transition要置入的行数
        index = self.memory_pointer % self.memory_size
        # 置入transition
        self.memory[index, :] = transition
        # memory_counter自加1
        self.memory_pointer += 1

    def choose_action(self, s):
        return self.actor(t.FloatTensor(s)).detach().numpy()

    def learn(self):
        # COPY PARAMETERS
        if self.t_replace_counter % self.replaycement_epoch == 0:
            self.t_replace_counter = 0
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
        # EXTRACT MEMORY
        batch_memory = self.sample_from_memory()

        b_s = t.FloatTensor(batch_memory[:, :self.n_features])
        b_a = t.FloatTensor(
            batch_memory[:, self.n_features:self.n_features+self.n_actions])
        b_r = t.FloatTensor(
            batch_memory[:, self.n_features+self.n_actions:self.n_features+self.n_actions+1])
        b_s_ = t.FloatTensor(
            batch_memory[:, self.n_features+self.n_actions+1:])

        # TRAIN ACTOR
        a = self.actor(b_s)
        q = self.critic(b_s, a)
        loss_actor = -t.mean(q)  # Q越大越好
        self.optimizer_actor.zero_grad()
        #################################################
        # 需要进行两次反向传播，所以采用retain_graph=True   #
        # 跟require_grad不同，require_grad=False的使用可以 #
        # 在训练过程中冻结网络参数 ，于此相同的还有with t.no_grad()#
        #################################################

        # loss_actor.backward(retain_graph=True) ?
        loss_actor.backward(retain_graph=False)
        self.optimizer_actor.step()

        # TRAIN CRITIC
        a_ = self.actor_target(b_s_)
        q_ = self.critic_target(b_s_, a_)
        q_target = b_r+self.reward_decay*q_
        q_eval = self.critic(b_s, b_a)
        td_error = self.loss_func(q_target, q_eval)
        self.optimizer_critic.zero_grad()
        td_error.backward()
        self.optimizer_critic.step()
        # UPDATE COUNTER
        self.t_replace_counter += 1
        return td_error.detach().item()

    def save(self):
        t.save(self.actor.state_dict(),
               "ACTOR {}-{}-{} {}-{}-{}.pth".format(time.localtime()[0],
                                                    time.localtime()[1],
                                                    time.localtime()[2],
                                                    time.localtime()[3],
                                                    time.localtime()[4],
                                                    time.localtime()[5], ))
        t.save(self.critic.state_dict(),
               "CRITIC{}-{}-{} {}-{}-{}.pth".format(time.localtime()[0],
                                                    time.localtime()[1],
                                                    time.localtime()[2],
                                                    time.localtime()[3],
                                                    time.localtime()[4],
                                                    time.localtime()[5], ))

    def load(self, path_a: str, path_c: str):
        model_a = t.load(path_a)
        model_c = t.load(path_c)
        self.actor.load_state_dict(model_a)
        self.critic.load_state_dict(model_c)
