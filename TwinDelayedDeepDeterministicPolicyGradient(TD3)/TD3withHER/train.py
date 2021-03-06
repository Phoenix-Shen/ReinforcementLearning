import numpy as np
import yaml
from model import Agent
import gym

if __name__ == "__main__":

    with open("TwinDelayedDeepDeterministicPolicyGradient(TD3)\TD3withHER\settings.yaml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), yaml.FullLoader)

    print("###################YOUR SETTINGS###################")
    for key in args.keys():
        print(f"[{key}]".ljust(30, " "), f"--->{args[key]}")
    print("->>>>training...")
    env = gym.make(args["env"])
    agent = Agent(
        env.observation_space.shape[0]*2,
        env.action_space.shape[0],
        args["hidden_size"],
        env.action_space.high,
        args["lr_a"],
        args["lr_c"],
        args["reward_decay"],
        args["buffer_size"],
        args["batch_size"],
        env,
        args["max_epoch"],
        args["log_dir"],
        args["tau"],
        args["policy_noise"],
        args["noise_clip"],
        args["update_policy_interval"],
        args["cuda"],
        args["display_interval"],
        args["model_save_dir"],
        args["save_frequency"],
        args["actor_dir"],
        args["critic_dir"],
        args["action_noise"],
        np.array([1, 0, 0], dtype=np.float32)
    )

    agent.learn()
    env.close()
