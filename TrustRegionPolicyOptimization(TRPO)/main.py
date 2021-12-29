import model
import arguments
import gym
if __name__ == "__main__":
    args = arguments.ARGS()
    env = gym.make(args.env_name)
    trpo_trainer = model.trpo_agent(env, args)
    trpo_trainer.learn()
    env.close()
