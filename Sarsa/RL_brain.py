from typing_extensions import ParamSpec
import numpy as np
import pandas as pd


class RL(object):
    def __init__(self, action_space, learning_rate=0.1, reward_decay=0.9, e_greedy=0.9) -> None:
        super().__init__()
        self.actions = action_space  # a list
        self.lr = learning_rate  # 学习率
        self.gamma = reward_decay   # 奖励衰减
        self.epsilon = e_greedy     # 贪婪度
        self.q_table = pd.DataFrame(
            columns=self.actions, dtype=np.float64)   # 初始 q_table

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def choose_action(self, observation):
        # 检测本 state 是否在 q_table 中存在(见后面标题内容)
        self.check_state_exist(observation)

        # 选择 action
        if np.random.uniform() < self.epsilon:  # 选择 Q value 最高的 action
            state_action = self.q_table.loc[observation, :]

            # 同一个 state, 可能会有多个相同的 Q action value, 所以我们乱序一下
            action = np.random.choice(
                state_action[state_action == np.max(state_action)].index)

        else:   # 随机选择 action
            action = np.random.choice(self.actions)

        return action

    def learn(self, *args):
        pass


class QLearningTable(RL):
    def __init__(self, action_space, learning_rate=0.1, reward_decay=0.9, e_greedy=0.9) -> None:
        super(QLearningTable, self).__init__(action_space,
                                             learning_rate=learning_rate, reward_decay=reward_decay, e_greedy=e_greedy)

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)


class SarsaTable(RL):
    def __init__(self, action_space, learning_rate=0.1, reward_decay=0.9, e_greedy=0.9) -> None:
        super(SarsaTable, self).__init__(action_space, learning_rate=learning_rate,
                                         reward_decay=reward_decay, e_greedy=e_greedy)

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            # q_target 基于选好的 a_ 而不是 Q(s_) 的最大值
            q_target = r + self.gamma * self.q_table.loc[s_, a_]
        else:
            q_target = r  # 如果 s_ 是终止符
        self.q_table.loc[s, a] += self.lr * \
            (q_target - q_predict)  # 更新 q_table


class SarsaLambdaTable(RL):
    def __init__(self, action_space, learning_rate=0.1, reward_decay=0.9, e_greedy=0.9, trace_decay=0.9) -> None:
        super(SarsaLambdaTable, self).__init__(action_space, learning_rate=learning_rate,
                                               reward_decay=reward_decay, e_greedy=e_greedy)
        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy()

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            to_be_append = pd.Series(
                [0] * len(self.actions),
                index=self.q_table.columns,
                name=state,
            )
            self.q_table = self.q_table.append(to_be_append)

            # also update eligibility trace
            self.eligibility_trace = self.eligibility_trace.append(
                to_be_append)

    def learn(self, s, a, r, s_, a_):
        # 这部分和 Sarsa 一样
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]
        else:
            q_target = r
        error = q_target - q_predict

        # 这里开始不同:
        # 对于经历过的 state-action, 我们让他+1, 证明他是得到 reward 路途中不可或缺的一环
        self.eligibility_trace.loc[s, a] += 1

        # Q table 更新
        self.q_table += self.lr * error * self.eligibility_trace

        # 随着时间衰减 eligibility trace 的值, 离获取 reward 越远的步, 他的"不可或缺性"越小
        self.eligibility_trace *= self.gamma*self.lambda_
