from models import Actor
import torch as t
import yaml
import gym
if __name__ == "__main__":
    with open("DistributedProximalPolicyOptimization(DPPO)\settings.yaml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    env = gym.make(args["env"])

    actor = Actor(
        env.observation_space.shape[0], env.action_space.shape[0], args["hidden_size"])
    total_reward = 0
    actor.load_state_dict(t.load(args["actor_dir"]))
    for ep in range(10):
        done = False
        obs = env.reset()
        reward_sum = 0
        while not done:
            env.render()
            with t.no_grad():
                obs_tensor = t.tensor(obs, ).unsqueeze(0)
                mean, _ = actor.forward(obs_tensor)
                actions = t.tanh(mean).cpu().detach().numpy()[0]
            obs_, reward, done, _ = env.step(actions)
            reward_sum += reward

            obs = obs_
        total_reward += reward_sum
        print(f"epoch {ep} finished, reward is {reward_sum  }")
    print("total episode:{},avg_reward:{}".format(10, total_reward/10))
