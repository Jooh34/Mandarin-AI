import numpy as np
import torch
import torch.nn as nn
import time
from collections import deque
from tqdm import tqdm
from torchsummary import summary
from multiprocessing import shared_memory
import threading

from core.board import Board
from ai.mcts import MCTS
from ai.file_manager import FileManager
from ai.config import AlphaZeroConfig

from core.types import Camp, MAX_ROW, MAX_COL

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
        self.nnet = None
        self.pi_list = []
    
    def train(self):
        file_manager = FileManager()
        replay_buffer = ReplayBuffer(self.config)
        self.nnet = file_manager.latest_network(replay_buffer)
        summary(self.nnet, (3, MAX_ROW, MAX_COL))

        epoch = 1
        while True:
            print(f'epoch : {epoch}')

            self.selfplay_game(self.nnet, replay_buffer, file_manager)

            self.train_network(replay_buffer, file_manager)
            epoch+=1

    def selfplay_game(self, nnet, replay_buffer: ReplayBuffer, file_manager: FileManager, save_replay=False):
        num_mcts = self.config.num_actors
        shared_input = np.zeros((num_mcts, 3, MAX_ROW, MAX_COL))  # neural net input

        # check all actors done
        done_list = [0]*num_mcts
        done_cnt = 0

        board_list = [Board() for _ in range(num_mcts)]
        board_history_list = [[] for _ in range(num_mcts)]
        pi_list = [[] for _ in range(num_mcts)]

        k = 1
        while done_cnt < num_mcts:
            print(f'{k}th move of {num_mcts} games.')
            k+=1

            mcts_list = []
            for i in range(num_mcts):
                board_history_list[i].append(board_list[i].get_board_state_to_evaluate())
                mcts = MCTS(self.config, board_list[i], shared_input, i)
                mcts_list.append(mcts)

                mcts.fill_shared_input()

            # first action
            policy_logits, value = nnet.inference(shared_input)
            for i,mcts in enumerate(mcts_list):
                mcts.after_inference(policy_logits[i][0], value[i])
                mcts.add_exploration_noise(mcts.root)
            ##

            ## mcts simulations
            for _ in range(self.config.num_simulations):
                for i,mcts in enumerate(mcts_list):
                    mcts.step(policy_logits, value)

                policy_logits, value = nnet.inference(shared_input)
                for i, mcts in enumerate(mcts_list):
                    mcts.after_inference(policy_logits[i][0], value[i])

            # select action
            for i, mcts in enumerate(mcts_list):
                action = mcts.select_action()
                board_list[i].take_action(action)
                pi = self.get_search_statistics(mcts.root)
                pi_list[i].append(pi)

                if board_list[i].is_terminal() and done_list[i] == 0:
                    done_list[i] = 1
                    done_cnt += 1
        
        # add to replay buffer
        for i in range(num_mcts):
            board_history = board_history_list[i]
            for j in range(len(board_history)):
                camp = Camp.Black if j%2 == 0 else Camp.White
                rw = board_list[i].get_terminal_value(camp)
                
                replay_buffer.append_reward_list(rw)
                replay_buffer.append_board_history(board_history_list[i][j])
                replay_buffer.append_pi_list(pi_list[i][j])

        print(f'{num_mcts} games terminated. replay_buffer size is {len(replay_buffer.reward_list)}')

    def train_network(self, replay_buffer: ReplayBuffer, file_manager: FileManager):
        nnet = self.nnet
        optimizer = torch.optim.AdamW(nnet.parameters(), lr=1e-1, weight_decay=self.config.weight_decay) # temp
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=0.001, max_lr=0.1, step_size_up=10, step_size_down=15, mode='triangular', cycle_momentum=False)
        

        train_start = time.time()
        # prevent overfitting when replay_buffer small
        _training_step = min(self.config.training_steps, int(len(replay_buffer.board_history)*5 // self.config.batch_size)+1)
        
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
        move_modality = [[[0]*1 for _ in range(MAX_COL)] for _ in range(MAX_ROW)]
        for action, child in root.children.items():
            move_modality[action[0]][action[1]][0] = child.visit_count / sum_visits

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