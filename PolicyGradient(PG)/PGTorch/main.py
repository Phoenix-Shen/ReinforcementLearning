import torch as t
import numpy as np
import gym
import matplotlib.pyplot as plt
from torch import optim
import torch
import torch.nn as nn
import torch.nn.functional as F

GAMMA = 0.99
RENDER = True
LR = 0.01
REWARD_DECAY = 0.95
LOG_INTERVAL = 10
env = gym.make("CartPole-v0").unwrapped
env.seed(1)


class PG(nn.Module):  # 继承torch.nn.Module
    def __init__(self, n_actions, n_features) -> None:
        super().__init__()  # 运行基类的初始化函数即nn.Module.__init__()
        self.affline1 = nn.Linear(n_actions, 128)  # self.xxxx即类内部的Public变量
        self.dropout = nn.Dropout(p=0.6)
        self.affline2 = nn.Linear(128, n_features)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affline1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affline2(x)
        return F.softmax(action_scores, dim=1)


def discount_and_norm_rewards(ep_rs):
    discounted_ep_rs = np.zeros_like(ep_rs)
    running_add = 0
    for t in reversed(range(0, len(ep_rs))):
        running_add = running_add * GAMMA + ep_rs[t]
        discounted_ep_rs[t] = running_add

    # normalize episode rewards
    discounted_ep_rs -= np.mean(discounted_ep_rs)
    discounted_ep_rs /= np.std(discounted_ep_rs)
    return torch.tensor(discounted_ep_rs)


def choose_action(net: PG, observation: list) -> None:
    observation_tensor_unsqueeze = t.unsqueeze(t.FloatTensor(observation), 0)
    # 注意 不能使用DETACH！！！！！！！！！！！！
    # 使用detach会导致梯度消失
    prob_weights = net(observation_tensor_unsqueeze)
    action = np.random.choice(
        range(prob_weights.detach().numpy().shape[1]), p=prob_weights.squeeze(dim=0).detach().numpy())
    # 记录当前动作
    net.saved_log_probs.append(prob_weights.log()[0][action].unsqueeze(dim=0))
    return action


def train_net(net: PG, optimizer: optim) -> None:

    R = 0
    policy_loss = []
    returns = []
    for r in net.rewards[::-1]:
        R = r + GAMMA * R
        returns.insert(0, R)        # 将R插入到指定的位置0处
    """
    returns 数组的最后形式应该是 [1.5.1.4...-1.5]类似这个样子
    这是为了提醒网络，多走前面几步，不要走使杆子立不起来的那几步，所以在前面多更新，后面少更新。
    """
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std())     # 归一化
    for log_prob, R in zip(net.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)          # 损失函数为交叉熵 nn.CrossEntropy

    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    # policy_loss.requires_grad_(True)         # 求和
    policy_loss.backward()

    # print(net.parameters())
    optimizer.step()
    del net.rewards[:]          # 清空episode 数据
    del net.saved_log_probs[:]
# 对于连续空间，我们可以用另外一种形式求梯度。


def mainLoop():

    # 定义网络
    pg = PG(4, 2)
    # 优化器
    optimizer = optim.Adam(pg.parameters(), lr=LR)

    running_reward = 10
    for i_episode in range(1000):        # 采集（训练）最多1000个序列
        state, ep_reward = env.reset(), 0    # ep_reward表示每个episode中的reward
        # print(state.shape)
        while True:
            action = choose_action(pg, state)
            state, reward, done, _ = env.step(action)
            if RENDER:
                env.render()
            pg.rewards.append(reward)
            ep_reward += reward
            if done:
                break
        running_reward = 0.05 * ep_reward + (1-0.05) * running_reward
        train_net(pg, optimizer=optimizer)
        if i_episode % LOG_INTERVAL == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward))
        if running_reward > env.spec.reward_threshold:   # 大于游戏的最大阈值475时，退出游戏
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == "__main__":
    mainLoop()
