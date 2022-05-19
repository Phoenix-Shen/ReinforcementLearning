import gym
import model
from tensorboardX import SummaryWriter
writer = SummaryWriter("./ProximalPolicyOptimization(PPO)/logs")
env = gym.make("LunarLanderContinuous-v2")
ppo = model.PPO(env, writer)
ppo.learn(1000000)
ppo.save_model("./ProximalPolicyOptimization/saved_models")
