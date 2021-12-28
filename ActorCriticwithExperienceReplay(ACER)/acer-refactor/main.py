import model


if __name__ == "__main__":
    agent = model.Agent(8, 4, 25, 16, 1e-3, 1500, "LunarLander-v2",render=False)
    agent.run()
