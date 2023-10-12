import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import os
from copy import deepcopy
from collections import deque

from core.board import Board, Formation
from ai.mcts import MCTS
from ai.mandarin_net import MandarinNet, ProxyUniformNetwork
from ai.config import AlphaZeroConfig
from core.move import Action
from core.types import Camp

BOARD_H = 10
BOARD_W = 9
BOARD_MOVE_MODAL = 59

class SharedStorage(object):
    def __init__(self, checkpoint_folder='checkpoint'):
        self.nnet = None
        abs_path = os.path.dirname(__file__)
        self.checkpoint_folder = os.path.join(abs_path, checkpoint_folder)

    def latest_network(self):
        lst = os.listdir(self.checkpoint_folder)
        candidate = []
        for file_name in lst:
            ext = file_name.split('.')[-1]
            if ext == '.pt':
                candidate.append(file_name)

        print(candidate)
        if candidate:
            mx = -1
            mx_name = ''
            for candi in candidate:
                num = int(candi.split('.')[0].split('_')[-1])
                if num > mx:
                    mx_name = candi
                    mx = num

            self.load_checkpoint(self.checkpoint_folder, mx_name)
            self.nnet.set_num_steps(mx)

            return self.nnet
        else:
            # return ProxyUniformNetwork()  # policy -> uniform, value -> 0.5
            self.nnet = MandarinNet()
            return self.nnet

    def save_checkpoint(self, num_steps):
        filename = f'mandarin_{num_steps}'

        # change extension
        if not os.path.exists(self.checkpoint_folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(self.checkpoint_folder))
            os.mkdir(self.checkpoint_folder)

        filename = f'mandarin_{num_steps}' + ".pt"
        filepath = os.path.join(self.checkpoint_folder, filename)
        torch.save(self.nnet, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='mandarin'):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))
        
        self.nnet = torch.load(filepath, map_location=device)

class ReplayBuffer(object):
    def __init__(self, config: AlphaZeroConfig):
        self.window_size = config.window_size
        self.batch_size = config.batch_size

        # for one game
        self.board_history = deque([])
        self.pi_list = deque([])
        self.reward_list = deque([])

        # real replay buffer
        self.buffer = deque([])
    
    def append_board_history(self, bh):
        if len(self.board_history) >= self.window_size:
            self.board_history.popleft()
        self.board_history.appendleft(bh)
    
    def append_pi_list(self, pi):
        if len(self.pi_list) >= self.window_size:
            self.pi_list.popleft()
        self.pi_list.appendleft(pi)
    
    def append_reward_list(self, r):
        if len(self.reward_list) >= self.window_size:
            self.reward_list.popleft()
        self.reward_list.appendleft(r)

    def sample_batch(self):
        # Sample uniformly
        length = len(self.buffer)
        idx_list = np.random.randint(length, size=self.batch_size)
        print(idx_list)

        return [self.buffer[i] for i in idx_list]
    
class ReplayDataset(data.Dataset):
    def __init__(self, replay_buffer: ReplayBuffer):
        super(ReplayDataset, self).__init__()

        self.board_history = replay_buffer.board_history
        self.pi_list = replay_buffer.pi_list
        self.reward_list = replay_buffer.reward_list
        
    def __getitem__(self, index):
        x = torch.as_tensor(self.board_history[index], dtype=torch.float)
        y = torch.as_tensor(self.pi_list[index], dtype=torch.float)
        z = torch.as_tensor([self.reward_list[index]], dtype=torch.float)
        return x,y,z

    def __len__(self):
        return len(self.pi_list)

class Trainer:
    def __init__(self):
        self.config = AlphaZeroConfig()
        self.mcts = MCTS(self.config)
        self.nnet = None
        self.pi_list = []
    
    def train(self):
        storage = SharedStorage()
        replay_buffer = ReplayBuffer(self.config)
        self.nnet = storage.latest_network()

        for i in range(self.config.n_games_to_train):
            self.run_selfplay(replay_buffer, storage)
            print(f'{i}-th game played')

        self.train_network(replay_buffer, storage)

    def run_selfplay(self, replay_buffer: ReplayBuffer, storage: SharedStorage):
        self.play_game(self.nnet, replay_buffer)

    def play_game(self, nnet, replay_buffer: ReplayBuffer):
        han_formation = Formation.get_random_formation()
        cho_formation = Formation.get_random_formation()
        board = Board(cho_formation, han_formation)

        while not board.is_terminal() and len(replay_buffer.board_history) < self.config.max_moves:
            replay_buffer.append_board_history(board.get_board_state_to_evaluate())

            # do mcts and take action
            action_id, root = self.mcts.run_mcts(board, nnet)
            board.take_action_by_id(action_id)

            pi = self.get_search_statistics(root)
            replay_buffer.append_pi_list(pi)

            self.mcts.show_timer(reset=True)
        
        # add reward
        for i in range(len(replay_buffer.board_history)):
            if i % 2 == 0: # CHO reward
                replay_buffer.append_reward_list(board.get_terminal_value(Camp.CHO))
            else:
                replay_buffer.append_reward_list(board.get_terminal_value(Camp.HAN))

        nnet.increase_num_steps()
        print(f'play_game terminated. {len(replay_buffer.board_history)} moves, winner : {board.winner}')

    def train_network(self, replay_buffer: ReplayBuffer, storage: SharedStorage):
        nnet = self.nnet
        optimizer = torch.optim.Adam(nnet.parameters(), lr=2e-1, weight_decay=self.config.weight_decay) # temp
        rds = ReplayDataset(replay_buffer)
        rdl = data.DataLoader(dataset=rds, batch_size=self.config.batch_size, num_workers=8, shuffle=True)

        for i in range(self.config.training_steps):
            if i % self.config.checkpoint_interval == 0:
                storage.save_checkpoint(i)
            
            self.update_weights(optimizer, nnet, rdl)

        storage.save_checkpoint(self.config.training_steps)
    
    def update_weights(self, optimizer, nnet, rdl):
        mse_loss = nn.MSELoss()
        for image, target_policy, target_value in rdl:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            image = image.to(device)
            target_policy = target_policy.flatten(start_dim=1).to(device)
            target_value = target_value.to(device)

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
