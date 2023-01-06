from model import PPOClip
import argparse
import yaml
from environment import make_env
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, help="your config file path",
                        default="./ProximalPolicyOptimization(PPO)/PPOGitHub/config.yaml")
    args = parser.parse_args()
    with open(args.config_file, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    print(config)
    env = make_env('CartPole-v1')
    algo = PPOClip(config)
    algo.learn(env)
