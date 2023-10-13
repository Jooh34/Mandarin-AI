TEST_MODE = 0
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
        if not TEST_MODE:
            # self.n_games_to_train = 2000
            self.n_games_to_train = 500

            self.num_simulations = 100

            ### Training
            self.training_steps = int(7e3)
            self.checkpoint_interval = int(1e3)
            self.window_size = int(1e6)
            self.batch_size = 4096

        else:
            self.n_games_to_train = 1
            self.num_simulations = 10

            ### Training
            self.training_steps = int(2)
            self.checkpoint_interval = int(1)
            self.window_size = int(1e3)
            self.batch_size = 32
