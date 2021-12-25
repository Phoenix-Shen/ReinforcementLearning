from model import Agent
import gym
import numpy as np

from tensorboardX import SummaryWriter
if __name__ == "__main__":
    writer = SummaryWriter("SoftActorCritic(SAC)\logs")
    env = gym.make("LunarLanderContinuous-v2")
    n_games = 1000
    agent = Agent(input_dims=env.observation_space.shape, env=env,
                  n_actions=env.action_space.shape[0])
    score_history = []
    global_step = 0
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0

        while not done:
            env.render()
            action = agent.choose_action(observation)
            observation_, reward, done, _ = env.step(action)
            score += reward
            agent.remember(observation, action,
                           reward, observation_, done)
            la, lc, lv = agent.learn()
            if la is not None:
                writer.add_scalar("actorLoss", la, global_step)
                writer.add_scalar("criticLoss", lc, global_step)
                writer.add_scalar("valueLoss", lv, global_step)
                global_step += 1
            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        writer.add_scalar("avg_score", avg_score, i)
        print("ep:{},score:{},avg_score:{}".format(i, score, avg_score))
