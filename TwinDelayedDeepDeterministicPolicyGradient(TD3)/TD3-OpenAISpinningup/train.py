import yaml
from models import Agent
if __name__ == "__main__":

    with open("TwinDelayedDeepDeterministicPolicyGradient(TD3)\TD3-OpenAISpinningup\settings.yaml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), yaml.FullLoader)

    print("###################YOUR SETTINGS###################")
    for key in args.keys():
        print(f"[{key}]".ljust(30, " "), f"--->{args[key]}")
    print("->>>>training...")

    agent = Agent(args)
    agent.learn()
