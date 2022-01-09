# %%
import yaml
from models import Agent
import gym

if __name__ == "__main__":
    with open("./SoftActorCritic(SAC)/SoftActorCriticWithoutValueNet/settings.yaml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    env = gym.make(args["env_name"])

    agent = Agent(
        env,
        env.observation_space.shape[0],
        env.action_space.shape[0],
        args["hiddenlayer_size"],
        args["gamma"],
        int(args["buffer_size"]),
        args["batch_size"],
        args["tau"],
        args["lr_critic"],
        args["lr_actor"],
        args["reward_scale"],
        args["log_std_max"],
        args["log_std_min"],
        env.action_space.high,
        args["entropy_weights"],
        args["init_exploration_steps"],
        args["n_episodes"],
        args["update_cycles"],
        args["target_update_interval"],
        args["eval_episodes"],
        args["eval_interval"],
        args["log_dir"],
        args["save_frequency"],
        args["save_dir"],
        args["cuda"],
    )

    agent.learn()

    env.close()
