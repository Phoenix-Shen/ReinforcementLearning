import gym
import numpy as np
import model
import tensorboardX
if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    n_features = env.observation_space.shape[0]
    n_actions = env.action_space.n
    writer = tensorboardX.SummaryWriter(
        "./ActorCriticwithExperienceReplay(ACER)/logs")
    n_games = 1500
    agent = model.Agent(n_features=8,
                        n_actions=n_actions,
                        reward_decay=0.99,
                        lr=1e-2,
                        batch_size=32,
                        mem_size=10000)

    scores = []
    total_step = 0
    for i in range(n_games):
        done = False
        observation = env.reset()
        score = 0

        while not done:
            action, prob = agent.choose_action(observation=observation)
            observation_, reward, done, _ = env.step(action)
            score += reward
            agent.store_transition(
                observation, prob, reward, observation_, int(done))

            actor_loss, critic_loss = agent.learn()
            if actor_loss is not None and critic_loss is not None:
                writer.add_scalar("actorloss", actor_loss, total_step)
                writer.add_scalar("criticloss", critic_loss, total_step)
                total_step += 1
            observation = observation_
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        writer.add_scalar("avg_score", avg_score, i)
        print("episode:{},score:{},AVG_score:{}".format(i, score, avg_score))
