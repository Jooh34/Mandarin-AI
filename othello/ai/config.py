MODE = 1
# 0 -> large
# 1 -> small
# 2 -> test

class AlphaZeroConfig(object):
    def __init__(self):
        self.max_moves = 100
        self.num_sampling_moves = 10

        # Root prior exploration noise.
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        self.weight_decay = 1e-4

        # discount factor : (additional)
        self.discount_factor = 0.7

        self.num_actors = 1024
        self.num_randomplay = 128


        if MODE == 0: # 10 hour plan
            # self.n_games_to_train = 2000
            self.n_games_to_train = 500

            self.num_simulations = 500

            ### Training
            self.training_steps = int(7e3)
            self.checkpoint_interval = int(1e3)
            self.window_size = int(1e6)
            self.batch_size = 1024

        elif MODE == 1: #
            self.num_simulations = 100

            ### Training
            self.training_steps = int(5e3)
            self.checkpoint_interval = int(1e3)
            self.window_size = int(5e5)
            self.batch_size = 1024

        else:
            self.n_games_to_train = 1
            self.num_simulations = 10

            ### Training
            self.training_steps = int(2)
            self.checkpoint_interval = int(1)
            self.window_size = int(1e6)
            self.batch_size = 1024
