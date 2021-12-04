from os import stat
import numpy as np
import pandas as pd
import time


"""
state  1 2 3 4 5
left   0 0 0 0 0
right  0 0 0 1 0
"""


np.random.seed(2)


N_STATES = 6   # 1维世界的宽度
ACTIONS = ['left', 'right']     # 探索者的可用动作
EPSILON = 0.9   # 贪婪度 greedy
ALPHA = 0.1     # 学习率
GAMMA = 0.9    # 奖励递减值
MAX_EPISODES = 13   # 最大回合数
FRESH_TIME = 0.3    # 移动间隔时间


def buildQTable(n_states: int, actions: list):
    table = pd.DataFrame(np.zeros((n_states, len(actions))),
                         columns=actions,
                         )
    print(table)
    return table


def chooseAction(state: int, q_table: pd.DataFrame) -> str:
    state_actions = q_table.iloc[state, :]
    if(np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        action_name = np.random.choice(ACTIONS)
    else:
        # state_actions.axes[state_actions.argmax()] 新版已经不适用了，应该换成idmax
        action_name = state_actions.idxmax()

    return action_name


def getEnvFeedback(S, A):
    if A == "right":
        if S == N_STATES-2:
            S_ = "terminal"
            R = 1
        else:
            S_ = S+1
            R = 0
    else:
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ = S-1

    return S_, R


def UpdateEnv(S, episode, step_counter):
    env_list = ["-"]*(N_STATES-1)+["T"]
    if S == "terminal":
        interaction = "Episode %s: total_steps=%s" % (episode+1, step_counter)
        print("\r{}".format(interaction), end="")
        time.sleep(2)
        print("\r                              ", end="")

    else:
        env_list[S] = 'O'
        interaction = "".join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def RLLoop():
    q_table = buildQTable(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        UpdateEnv(S, episode, step_counter)
        while not is_terminated:

            A = chooseAction(S, q_table=q_table)
            #print("\n {}".format(type(A)))
            S_, R = getEnvFeedback(S, A)
            q_predict = q_table.loc[S, A]
            if S_ != "terminal":
                q_target = R+GAMMA*q_table.iloc[S_, :].max()
            else:
                q_target = R
                is_terminated = True
            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # q_table 更新
            S = S_  # 探索者移动到下一个 state

            UpdateEnv(S, episode, step_counter+1)  # 环境更新

            step_counter += 1
    return q_table


if __name__ == "__main__":
    q_table = RLLoop()
    print('\r\nQ-table:\n')
    print(q_table)
