class ARGS:
    def __init__(self) -> None:
        self.gamma = 0.99
        self.env_name = "LunarLanderContinuous-v2"
        self.seed = 123
        self.save_dir = "saved_models/"
        self.total_timesteps = 1e6
        self.nsteps = 1024
        self.lr = 3e-4
        self.batch_size = 64
        self.vf_itrs = 5
        self.tau = 0.95
        self.damping = 0.1
        self.max_kl = 0.01
        self.cuda = True
        self.env_type = "mojuco"
        self.log_dir = "logs"
