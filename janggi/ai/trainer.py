import numpy as np
import torch
import torch.nn as nn
import time
from collections import deque
from tqdm import tqdm

from core.board import Board, Formation
from ai.mcts import MCTS
from ai.file_manager import FileManager
from ai.config import AlphaZeroConfig
from core.move import Action
from core.types import Camp

BOARD_H = 10
BOARD_W = 9
BOARD_MOVE_MODAL = 59

class ReplayBuffer(object):
    def __init__(self, config: AlphaZeroConfig):
        self.window_size = config.window_size
        self.batch_size = config.batch_size

        # for one game
        self.board_history = deque([])
        self.pi_list = deque([])
        self.reward_list = deque([])
    
    def load_from_pickle(self, data):
        if data.get('board_history'):
            self.board_history = deque(data['board_history'])
        if data.get('pi_list'):
            self.pi_list = deque(data['pi_list'])
        if data.get('reward_list'):
            self.reward_list = deque(data['reward_list'])

    def append_board_history(self, bh):
        if len(self.board_history) >= self.window_size:
            self.board_history.popleft()
        self.board_history.append(bh)
    
    def append_pi_list(self, pi):
        if len(self.pi_list) >= self.window_size:
            self.pi_list.popleft()
        self.pi_list.append(pi)
    
    def append_reward_list(self, r):
        if len(self.reward_list) >= self.window_size:
            self.reward_list.popleft()
        self.reward_list.append(r)

    def sample_batch(self):
        # Sample uniformly
        length = len(self.board_history)

        np_bh = np.array(self.board_history)
        np_pi_list = np.array(self.pi_list)
        np_reward_list = np.array(self.reward_list)

        rd_idx_list = [np.random.randint(length) for _ in range(self.batch_size)]
        return np_bh[rd_idx_list, :, :, :], np_pi_list[rd_idx_list, :, :, :], np_reward_list[rd_idx_list].reshape((self.batch_size, 1))

class Trainer:
    def __init__(self):
        self.config = AlphaZeroConfig()
        self.mcts = MCTS(self.config)
        self.nnet = None
        self.pi_list = []
    
    def train(self):
        file_manager = FileManager()
        replay_buffer = ReplayBuffer(self.config)
        self.nnet = file_manager.latest_network(replay_buffer)

        epoch = 1
        while True:
            print(f'epoch : {epoch}')
            for i in range(self.config.n_games_to_train):
                print(f'{i+1}-th game playing')
                self.selfplay_game(self.nnet, replay_buffer, file_manager, True if i<5 else False)

            self.train_network(replay_buffer, file_manager)
            epoch+=1

    def selfplay_game(self, nnet, replay_buffer: ReplayBuffer, file_manager: FileManager, save_replay=False):
        han_formation = Formation.get_random_formation()
        cho_formation = Formation.get_random_formation()
        board = Board(cho_formation, han_formation)

        board_history = []
        pi_list = []
        if save_replay:
            action_history = []
        while not board.is_terminal() and len(board_history) < self.config.max_moves:
            board_history.append(board.get_board_state_to_evaluate())

            # do mcts and take action
            action_id, root = self.mcts.run_mcts(board, nnet)
            if save_replay:
                gibo_str = Action.init_by_id(action_id).to_gibo_str(turn=board.turn)
                action_history.append(gibo_str)

            board.take_action_by_id(action_id)

            # save data for training
            pi = self.get_search_statistics(root)
            pi_list.append(pi)
        
        if save_replay:
            file_manager.save_replay(action_history, nnet.num_steps, self.config.num_simulations, cho_formation, han_formation)

        # add to replay buffer
        for i in range(len(board_history)):
            rev_i = len(board_history)-i

            rw = 0
            if i % 2 == 0: # CHO reward
                rw = board.get_terminal_value(Camp.CHO) * 10
            else:
                rw = board.get_terminal_value(Camp.HAN) * 10
            
            discounted_rw = self.get_discounted_reward(rev_i, rw)
            
            replay_buffer.append_reward_list(discounted_rw)
            replay_buffer.append_board_history(board_history[i])
            replay_buffer.append_pi_list(pi_list[i])

        print(f'play_game terminated. {len(board_history)} moves, winner : {board.winner}')

    def train_network(self, replay_buffer: ReplayBuffer, file_manager: FileManager):
        nnet = self.nnet
        optimizer = torch.optim.AdamW(nnet.parameters(), lr=1e-1, weight_decay=self.config.weight_decay) # temp
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=0.001, max_lr=0.1, step_size_up=10, step_size_down=15, mode='triangular', cycle_momentum=False)
        

        train_start = time.time()
        # prevent overfitting when replay_buffer small
        _training_step = min(self.config.training_steps, int(len(replay_buffer.board_history) // self.config.batch_size))
        
        print(f'training network.. step to train is {_training_step}')
        file_manager.save_replay_buffer(replay_buffer)
        for i in tqdm(range(_training_step)):
            if i % self.config.checkpoint_interval == 0:
                file_manager.save_checkpoint()
            
            batch = replay_buffer.sample_batch()
            self.update_weights(optimizer, nnet, batch)
            nnet.increase_num_steps()
            scheduler.step()

        elapsed = time.time()-train_start
        print(f'finished training network. elapsed {elapsed} seconds')
        file_manager.save_checkpoint()
        file_manager.save_replay_buffer(replay_buffer)
        
    
    def update_weights(self, optimizer, nnet, batch):
        mse_loss = nn.MSELoss()
        
        image, target_policy, target_value = batch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        image = torch.Tensor(image).to(device)
        target_policy = torch.Tensor(target_policy).flatten(start_dim=1).to(device)
        target_value = torch.Tensor(target_value).to(device)

        policy_logits, value = nnet(image)
        loss = (
            mse_loss(value, target_value) +
            nn.functional.cross_entropy(policy_logits, target_policy)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
    def get_search_statistics(self, root):
        # pi history for training
        sum_visits = sum(child.visit_count for child in root.children.values())
        move_modality = [[[0]*BOARD_MOVE_MODAL for _ in range(BOARD_W)] for _ in range(BOARD_H)]
        for action_id, child in root.children.items():
            action = Action.init_by_id(action_id)
            move_modality[action.prev[0]][action.prev[1]][action.move_type] = child.visit_count / sum_visits

        return move_modality

    def get_discounted_reward(self, num_step, reward):
        df = self.config.discount_factor
        if df >= 1:
            return reward
        else:
            partial = 1
            if num_step >= 10:
                partial = df**10
            else:
                partial = df**num_step
            
            return reward * partial