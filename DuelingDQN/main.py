import models
import gym


def generate_env(env_name: str):
    env = gym.make(env_name)
    env.seed(1)  # reproducible
    env = env.unwrapped
    N_F = env.observation_space.shape[0]
    N_A = env.action_space.n
    return env,  N_F, N_A


env, n_states, n_actions = generate_env('MountainCar-v0')
memory_size = 50000
save_frequency = 20
dqn = models.DuelingDQN(n_states, n_actions,
                        hidden_layers=512, memory_size=memory_size)
total_steps = 0
print('\nCollecting experience...')
for i_episode in range(1, 400):
    s = env.reset()

    ep_r = 0

    while True:
        env.render()
        a = dqn.choose_action(s)

        s_, reward, done, _ = env.step(a)

        if done:
            reward = 10
        ep_r += reward

        dqn.store_transition(s, a, reward, s_)

        if total_steps > memory_size:
            dqn.learn()
            print("training->episode:{},ep_r:{}".format(i_episode, ep_r))
        if done:
            ep_r = 0
            s = env.reset()
            break

        s = s_
        total_steps += 1
    if i_episode % save_frequency == 0:
        dqn.save()
