import torch as t
import model
import gym
import numpy as np
from tensorboardX import SummaryWriter
if __name__ == "__main__":
    model_dir = "./ProximalPolicyOptimization(PPO)/PPOMedium/saved_models/ACTOR 2022-5-20 15-54-15.pth"
    writer = SummaryWriter("./ProximalPolicyOptimization(PPO)/PPOMedium/logs")
    env = gym.make("Pendulum-v1")
    ppo = model.PPO(env, writer)
    ppo.actor.load_state_dict(t.load(model_dir))
    done = False
    ep_r = []
    game_id = 0
    s = env.reset()
    env.render()
    while True:
        env.render()
        a = ppo.actor.forward(t.FloatTensor(s)).detach().numpy()

        s_, r, done, _ = env.step(a)
        s = s_
        print("action:{},reward:{}".format(a, r))

        if done:
            s = env.reset()
