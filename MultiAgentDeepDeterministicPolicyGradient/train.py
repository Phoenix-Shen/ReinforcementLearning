from model import MADDPG
from utils import load_cfg
from pursuit import MAWaterWorld_mod

if __name__ == "__main__":
    args = load_cfg(r"MultiAgentDeepDeterministicPolicyGradient\config.yaml")

    food_reward = 10.0
    poison_reward = -1.0
    encounter_reward = 0.01
    n_coop = 2
    env = MAWaterWorld_mod(
        n_pursuers=2,
        n_evaders=50,
        n_poison=50,
        obstacle_radius=0.04,
        food_reward=food_reward,
        poison_reward=poison_reward,
        encounter_reward=encounter_reward,
        n_coop=n_coop,
        sensor_range=0.2,
        obstacle_loc=None,
    )

    agent = MADDPG(
        args["n_agents"],
        args["obs_dim"],
        args["action_dim"],
        args["batch_size"],
        args["capacity"],
        args["n_explore"],
        args["n_episodes"],
        args["reward_decay"],
        args["tau"],
        args["lr"],
        args["cuda"],
        env,
        args["max_steps"],
        args["reward_scale"],
        args["soft_update_interval"],
        args["seed"],
    )

