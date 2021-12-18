import model
import gym
env = gym.make("LunarLander-v2")
agent = model.ActorCritic(8, 3, env)
agent.learn()
