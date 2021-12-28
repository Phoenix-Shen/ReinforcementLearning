import model


if __name__ == "__main__":
    agent = model.Agent(8, 4, 25, 16, 1e-5, 1500, "LunarLander-v2")
    agent.run()
