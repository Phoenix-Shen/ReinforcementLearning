import gym
import matplotlib.pyplot as plt
from RL_brain import PolicyGradient
RENDER=False
DISPLAY_REWARD_THRESHOLD=400

env=gym.make("CartPole-v0").unwrapped
env.seed(1)

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

PG=PolicyGradient(2,4,learning_rate=0.02,reward_decay=0.99)

for i_episode in range(3000):
    observation=env.reset()
    
    while True:
        action=PG.choose_action(observation=observation)
        observation_, reward, done, info = env.step(action)
        PG.store_transition(observation, action, reward)  
        if done:
            ep_rs_sum = sum(PG.ep_rs)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # 判断是否显示模拟
            print("episode:", i_episode, "  reward:", int(running_reward))

            vt = PG.learn() # 学习, 输出 vt, 我们下节课讲这个 vt 的作用

            if i_episode == 0:
                plt.plot(vt)    # plot 这个回合的 vt
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()
            break

        observation = observation_
