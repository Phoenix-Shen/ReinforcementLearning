from itertools import count
from model import Agent
from collections import Counter, deque
import gym
import numpy as np


def SAC(n_ep: int, env: gym.Env, agent: Agent, render: bool):
    scores_deque = deque(maxlen=100)
    avg_100_scores = []
    for i_episode in range(1, n_ep+1):

        done = False
        state = env.reset()
        state = state.reshape((1, env.observation_space.shape[0]))
        score = 0
        for t in count(0):
            if render:
                env.render()
            action = agent.act(state)
            action_v = action.numpy()
            action_v = np.clip(action_v*env.action_space.high[0],
                               env.action_space.low[0], env.action_space.high[0])
            next_state, reward, done, _ = env.step(action_v)
            next_state = next_state.reshape(
                (1, env.observation_space.shape[0]))
            agent.step(state, action, reward, next_state, done, t)
            state = next_state
            score += reward
            if done:
                break
        scores_deque.append(score)
        avg_100_scores.append(np.mean(scores_deque))
        print('\rEpisode {} Reward: {:.2f}  Average100 Score: {:.2f}'.format(
            i_episode, score, np.mean(scores_deque)), end="")
        if i_episode % 10 == 0:
            print('\rEpisode {}  Reward: {:.2f}  Average100 Score: {:.2f}'.format(
                i_episode, score, np.mean(scores_deque)))


if __name__ == "__main__":
    env = gym.make("LunarLanderContinuous-v2")
    agent = Agent(8, 2)
    SAC(1000, env, agent, False)
