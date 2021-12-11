import numpy as np
from model import DDPG
import gym
import time


def generate_env(env_name: str):
    env = gym.make(env_name)
    env.seed(1)  # reproducible
    env = env.unwrapped
    N_F = env.observation_space.shape[0]
    N_A = env.action_space
    return env,  N_F, N_A


# HYPER PARAMETERS
env, N_states, _ = generate_env("Pendulum-v1")
MAX_EPISODES = 200
MAX_EP_STEPS = 1000
MEMORY_CAPACITY = 50000
RENDER = False
STD = 3.0
LR_A = 1e-4
LR_C = 2e-4
# TRAIN
ddpg = DDPG(N_states, 1, MEMORY_CAPACITY, 100, lr_a=LR_A, lr_c=LR_C)


for i in range(MAX_EPISODES):
    t1 = time.time()
    s = env.reset()
    ep_reward = 0
    td_errors = []
    for step in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        # Add exploration noise
        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, STD), -2, 2)

        s_, r, done, _ = env.step(a)

        ddpg.store_transition(s, a, r/10, s_)

        if ddpg.memory_pointer > MEMORY_CAPACITY:
            STD = STD*0.995
            td_error = ddpg.learn()
            td_errors.append(td_error)
        s = s_
        ep_reward = ep_reward+r

    """if ep_reward > -300:
        RENDER = True"""
    t2 = time.time()

    print("Episode:{},Reward:{},TD_error:{},UseTime:{}".format(
        i, ep_reward, np.mean(td_errors), t2-t1))
ddpg.save()
