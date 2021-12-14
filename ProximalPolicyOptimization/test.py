import torch as t
import model
import gym
import numpy as np
from tensorboardX import SummaryWriter
if __name__ == "__main__":
    model_dir = "./ProximalPolicyOptimization/ACTOR 2021-12-14 16-45-2.pth"
    writer = SummaryWriter("./ProximalPolicyOptimization/logs")
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

        s_, r, _, _ = env.step(a)
        s = s_
        print("action:{},reward:{}".format(a, r))
