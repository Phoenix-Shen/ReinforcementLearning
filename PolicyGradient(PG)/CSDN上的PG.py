# 策略梯度算法
# 2020.5.22
#
# cartpole 的state是一个4维向量，分别是位置，速度，杆子的角度，加速度；action是二维、离散，即向左/右推杆子
# 每一步的reward都是1  游戏的threshold是475

import argparse
import numpy as np
import gym
from itertools import count
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

parser = argparse.ArgumentParser(description='Pytorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99,
                    metavar='G', help='discount factor(default:0.99)')
parser.add_argument('--seed', type=int, default=543,
                    metavar='N', help='random seed (default: 543)')
parser.add_argument('--render', action='store_false',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('CartPole-v1')
env.seed(args.seed)
torch.manual_seed(args.seed)    # 策略梯度算法方差很大，设置seed以保证复现性
print('observation space:', env.observation_space)
print('action space:', env.action_space)


class Policy(nn.Module):
    # 离散空间采用了 softmax policy 来参数化策略
    def __init__(self):
        super(Policy, self).__init__()
        self.affline1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affline2 = nn.Linear(128, 2)  # 两种动作

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affline1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affline2(x)
        return F.softmax(action_scores, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()  # 非负的最小值，使得归一化时分母不为0


def select_action(state):
    # 选择动作，这个动作不是根据Q值来选择，而是使用softmax生成的概率来选
    #  不需要epsilon-greedy，因为概率本身就具有随机性
    state = torch.from_numpy(state).float().unsqueeze(0)
    # print(state.shape)   torch.size([1,4])
    probs = policy(state)
    # print(probs)
    # print(probs.log())
    m = Categorical(probs)      # 生成分布
    action = m.sample()           # 从分布中采样
    # print(m.log_prob(action))   # m.log_prob(action)相当于probs.log()[0][action.item()].unsqueeze(0)
    policy.saved_log_probs.append(m.log_prob(action))    # 取对数似然 logπ(s,a)
    return action.item()         # 返回一个元素值


def finish_episode():
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)        # 将R插入到指定的位置0处
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)     # 归一化
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)          # 损失函数为交叉熵
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()          # 求和
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]          # 清空episode 数据
    del policy.saved_log_probs[:]


def main():
    running_reward = 10
    for i_episode in range(1000):        # 采集（训练）最多1000个序列
        state, ep_reward = env.reset(), 0    # ep_reward表示每个episode中的reward
        # print(state.shape)
        for t in range(1, 1000):
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1-0.05) * running_reward
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward))
        if running_reward > env.spec.reward_threshold:   # 大于游戏的最大阈值475时，退出游戏
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()
