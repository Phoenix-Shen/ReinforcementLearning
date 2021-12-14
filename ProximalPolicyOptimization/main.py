import gym
import model
from tensorboardX import SummaryWriter
writer = SummaryWriter("./ProximalPolicyOptimization/logs")
env = gym.make("Pendulum-v1")
ppo = model.PPO(env, writer)
ppo.learn(100000)
