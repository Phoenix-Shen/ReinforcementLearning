from model import Agent
import gym
import numpy as np

from tensorboardX import SummaryWriter
if __name__ == "__main__":
    writer = SummaryWriter("SoftActorCritic(SAC)\logs")
    env = gym.make("LunarLanderContinuous-v2")
    n_games = 1000
    # 1. Initialize Parameters theta omega phi and phi'
    agent = Agent(input_dims=env.observation_space.shape, env=env,
                  n_actions=env.action_space.shape[0])
    score_history = []
    global_step = 0
    # 2. for each iteration do:
    # 3. in practice, a combination of a single environment step and multiple gradient steps is found to work best
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        # 4. for each environment steup do:
        while not done:
            env.render()
            # 5. at~pi_theta(at|st)
            action = agent.choose_action(observation)
            # 6. s_t+1~rou_pi(s_t+1|s_t,a_t)
            observation_, reward, done, _ = env.step(action)
            score += reward
            # 7. D←--D∪{(st,at,r(st,at),st+1)}
            agent.remember(observation, action,
                           reward, observation_, done)
            # 8. for each gradient update step do
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
