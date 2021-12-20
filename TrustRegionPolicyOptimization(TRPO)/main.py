
import agent


trpo = agent.AGENT(8, 4, 128, 128, 150, "LunarLander-v2")
trpo.learn()
