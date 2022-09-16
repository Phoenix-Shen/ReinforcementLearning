import yaml
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios


def load_cfg(filename: str) -> dict:
    with open(filename, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), yaml.FullLoader)

    print("###################YOUR SETTINGS###################")
    for key in args.keys():
        print(f"[{key}]".ljust(30, " "), f"--->{args[key]}")
    return args


def make_env(args: dict):
    scenario = scenarios.load(args["scenario_name"] + ".py").Scenario()

    world = scenario.make_world()

    env = MultiAgentEnv(
        world, scenario.reset_world, scenario.reward, scenario.observation
    )

    args["n_players"] = env.n  # 包含敌人的所有玩家个数
    args["n_agents"] = env.n - args["num_advs"]  # 需要操控的玩家个数，虽然敌人也可以控制，但是双方都学习的话需要不同的算法
    args["obs_shape"] = [
        env.observation_space[i].shape[0] for i in range(args["n_agents"])
    ]  # 每一维代表该agent的obs维度
    action_shape = []
    for content in env.action_space:
        action_shape.append(content.n)
    args["action_shape"] = action_shape[: args["n_agents"]]  # 每一维代表该agent的act维度
    args["high_action"] = 1
    return env, args
