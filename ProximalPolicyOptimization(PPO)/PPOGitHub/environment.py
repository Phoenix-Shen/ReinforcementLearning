import gym


def make_env(env_name: str):
    env = gym.make(env_name)
    return env
