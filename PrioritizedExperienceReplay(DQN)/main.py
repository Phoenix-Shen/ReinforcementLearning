from model import DQNwithPER

import gym


# 使用gym库中的环境：CartPole，且打开封装
env = gym.make('CartPole-v0').unwrapped
N_ACTIONS = env.action_space.n                  # 杆子动作个数 (2个)
N_STATES = env.observation_space.shape[0]  # 杆子状态个数 (4个)
MEMORY_CAPACITY = 50  # 能够记忆100000个情况
dqn = DQNwithPER(N_STATES, N_ACTIONS, memory_size=MEMORY_CAPACITY)

total_steps = 0
for i in range(400):                                                    # 400个episode循环
    print('<<<<<<<<<Episode: %s' % i)
    s = env.reset()                                                     # 重置环境
    # 初始化该循环对应的episode的总奖励
    episode_reward_sum = 0

    # 开始一个episode (每一个循环代表一步)
    while True:
        env.render()                                                    # 显示实验动画
        # 输入该步对应的状态s，选择动作
        a = dqn.choose_action(s)
        # 执行动作，获得反馈
        s_, r, done, info = env.step(a)

        # 修改奖励 (不修改也可以，修改奖励只是为了更快地得到训练好的摆杆)
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / \
            env.theta_threshold_radians - 0.5
        new_r = r1 + r2

        dqn.store_transition(s, a, new_r, s_)                 # 存储样本
        # 逐步加上一个episode内每个step的reward
        episode_reward_sum += new_r

        s = s_                                                # 更新状态
        total_steps += 1
        print(total_steps)
        if total_steps > MEMORY_CAPACITY:
            print("func learn")              # 如果累计的transition数量超过了记忆库的固定容量2000
            dqn.learn()  # 开始学习 (抽取记忆，即32个transition，并对评估网络参数进行更新，并在开始学习后每隔100次将评估网络的参数赋给目标网络)

        if done:       # 如果done为True
            # round()方法返回episode_reward_sum的小数点四舍五入到2个数字
            print('episode%s---reward_sum: %s' %
                  (i, round(episode_reward_sum, 2)))
            break                                             # 该episode结束
