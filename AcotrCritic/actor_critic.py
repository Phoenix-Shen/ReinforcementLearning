import gym
import models


def generate_env(env_name: str):
    env = gym.make(env_name)
    env.seed(1)  # reproducible
    env = env.unwrapped
    N_F = env.observation_space.shape[0]
    N_A = env.action_space.n
    return {"env": env, "features": N_F, "actions": N_A}


def main():
    MAX_EPISODE = 3000
    DISPLAY_REWARD_THRESHOLD = 200
    MAX_EP_STEPS = 1000
    RENDER = False
    GAMMA = 0.9
    LR_A = 0.001
    LR_C = 0.01
    env_property = generate_env('CartPole-v0')
    env = env_property["env"]
    actor = models.Actor(
        env_property["features"], env_property["actions"], lr=LR_A)
    critic = models.Critic(env_property["features"], lr=LR_C, GAMMA=GAMMA)
    for i_episode in range(MAX_EPISODE):
        s = env.reset()
        t = 0
        track_r = []
        while True:
            if RENDER:
                env.render()

            a = actor.choose_action(s)

            s_, r, done, _ = env.step(a)

            if done:
                r = -20

            track_r.append(r)

            # gradient = grad[r + gamma * V(s_) - V(s)]
            td_error = critic.learn(s, r, s_)
            # true_gradient = grad[logPi(s,a) * td_error]
            actor.learn(td_error)

            s = s_
            t += 1

            if done or t >= MAX_EP_STEPS:
                ep_rs_sum = sum(track_r)

                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
                if running_reward > DISPLAY_REWARD_THRESHOLD:
                    RENDER = True  # rendering
                print("episode:", i_episode, "  reward:", int(running_reward))
                break


if __name__ == "__main__":

    main()
