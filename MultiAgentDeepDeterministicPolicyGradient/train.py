from models import MADDPG
from utils import load_cfg, make_env


if __name__ == "__main__":
    args = load_cfg(r"MultiAgentDeepDeterministicPolicyGradient/config.yaml")
    env, args = make_env(args)

    agent = MADDPG(
        args["noise_rate"],
        args["epsilon"],
        args["max_episode_length"],
        env,
        args["n_agents"],
        args["n_players"],
        args["save_dir"],
        args["action_shape"],
        args["obs_shape"],
        args["high_action"],
        args["lr_actor"],
        args["lr_critic"],
        args["log_dir"],
        args["time_steps"],
        args["buffer_size"],
        args["batch_size"],
        args["gamma"],
        args["tau"],
        args["eval_interval"],
        args["eval_episodes"],
        args["eval_episodes_len"],
        args["cuda"],
    )

    agent.learn()
