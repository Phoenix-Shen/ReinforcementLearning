import gym
import model
from tensorboardX import SummaryWriter
writer = SummaryWriter("./ProximalPolicyOptimization(PPO)/PPOMedium/logs")
env = gym.make("Pendulum-v1")
ppo = model.PPO(env, writer)
ppo.learn(1500000)
ppo.save_model("./ProximalPolicyOptimization(PPO)/PPOMedium/saved_models")
