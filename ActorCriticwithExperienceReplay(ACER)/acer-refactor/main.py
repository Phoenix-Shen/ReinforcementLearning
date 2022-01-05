import model


if __name__ == "__main__":
    agent = model.Agent(8, 4, 100, 32, 0.001, 1500,
                        "LunarLander-v2", render=False)
    agent.run()
