MODE = 1
# 0 -> large
# 1 -> small
# 2 -> test

class AlphaZeroConfig(object):
    def __init__(self):
        self.max_moves = 200
        self.num_sampling_moves = 30

        # Root prior exploration noise.
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        self.weight_decay = 1e-4
        if MODE == 0:
            # self.n_games_to_train = 2000
            self.n_games_to_train = 300

            self.num_simulations = 50

            ### Training
            self.training_steps = int(7e3)
            self.checkpoint_interval = int(1e3)
            self.window_size = int(1e6)
            self.batch_size = 1024

        elif MODE == 1:
            self.n_games_to_train = 30
            self.num_simulations = 50

            ### Training
            self.training_steps = int(7e2)
            self.checkpoint_interval = int(1e2)
            self.window_size = int(1e5)
            self.batch_size = 1024

        else:
            self.n_games_to_train = 1
            self.num_simulations = 10

            ### Training
            self.training_steps = int(2)
            self.checkpoint_interval = int(1)
            self.window_size = int(1e3)
            self.batch_size = 1024
