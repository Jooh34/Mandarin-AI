class AlphaZeroConfig(object):
  def __init__(self):
    ### Self-Play
    # self.num_actors = 5000
    # self.n_games_to_train = 2000
    self.n_games_to_train = 100

    self.num_sampling_moves = 30
    self.max_moves = 200
    # self.num_simulations = 100
    self.num_simulations = 50

    # Root prior exploration noise.
    self.root_dirichlet_alpha = 0.3  # for chess, 0.03 for Go and 0.15 for shogi.
    self.root_exploration_fraction = 0.25

    # UCB formula
    self.pb_c_base = 19652
    self.pb_c_init = 1.25

    ### Training
    self.training_steps = int(700e3)
    self.checkpoint_interval = int(1e3)
    self.window_size = int(1e6)
    self.batch_size = 4096
    # self.training_steps = int(1)
    # self.checkpoint_interval = int(1)
    # self.window_size = int(10)
    # self.batch_size = 32

    self.weight_decay = 1e-4
    self.momentum = 0.9
    # Schedule for chess and shogi, Go starts at 2e-2 immediately.
    self.learning_rate_schedule = {
        0: 2e-1,
        100e3: 2e-2,
        300e3: 2e-3,
        500e3: 2e-4
    }