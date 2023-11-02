import numpy as np
import torch
import torch.nn as nn
import time
from collections import deque
from tqdm import tqdm
from torchsummary import summary
import random

from core.board import Board
from ai.mcts import MCTS
from ai.file_manager import FileManager
from ai.config import AlphaZeroConfig
import matplotlib.pyplot as plt

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

        self.plot_data = [[],[]]
        self.winrate_data = [[],[]]
    
    def train(self):
        file_manager = FileManager()
        replay_buffer = ReplayBuffer(self.config)
        self.nnet = file_manager.latest_network(replay_buffer)
        # summary(self.nnet, (3, MAX_ROW, MAX_COL))

        epoch = 1
        while True:
            print(f'epoch : {epoch}')

            self.selfplay_game(self.nnet, replay_buffer, file_manager)
            self.train_network(replay_buffer, file_manager)
            self.evaluate_vs_randomplay(self.nnet, file_manager, 100)
            self.save_plot_image(self.nnet.num_steps)

            # self.selfplay_game(self.nnet, replay_buffer, file_manager)
            epoch+=1

    def make_replay(self):
        file_manager = FileManager()
        replay_buffer = ReplayBuffer(self.config)
        nnet = file_manager.latest_network(replay_buffer)
        
        for i in range(10):
            board = Board()
            action_history = []
            next_root_node = None
            while not board.is_terminal():
                mcts = MCTS(self.config, board, None, None, next_root_node)
                action_probabilities = mcts.run_mcts(board, nnet, self.config.num_simulations, False) # sorted reverse
                action = action_probabilities[0][2]
                action_history.append(action)
                board.take_action(action)

                next_root_node = mcts.root.children[action]
            
            file_manager.save_replay(action_history, nnet.num_steps, self.config.num_simulations, board.winner)
            print(f'replay {i} generated.')

    def evaluate_vs_randomplay(self, nnet, file_manager: FileManager, num_simulation):
        num_mcts = self.config.num_randomplay
        shared_input = np.zeros((num_mcts, 3, MAX_ROW, MAX_COL))  # neural net input

        # check all actors done
        done_list = [0]*num_mcts
        done_cnt = 0

        board_list = [Board() for _ in range(num_mcts)]
        action_history = [[] for _ in range(num_mcts)]

        k = 0
        while done_cnt < num_mcts:
            print(f'{k}th move of {num_mcts} games.')

            if k % 2 == 0: # random-player always Black
                for i in range(num_mcts):
                    actions = board_list[i].get_possible_actions(board_list[i].turn)
                    action = random.choice(actions)
                    action_history[i].append(action)
                    board_list[i].take_action(action)
            
            else: # ----  ai-play ----
                mcts_list = []
                for i in range(num_mcts):
                    mcts = MCTS(self.config, board_list[i], shared_input, i)
                    mcts_list.append(mcts)

                    mcts.fill_shared_input()

                # first action
                policy_logits, value = nnet.inference(shared_input)
                for i,mcts in enumerate(mcts_list):
                    mcts.after_inference(policy_logits[i][0], value[i])
                    # mcts.add_exploration_noise(mcts.root)
                ##

                ## mcts simulations
                for _ in range(num_simulation):
                    for i,mcts in enumerate(mcts_list):
                        mcts.step(policy_logits, value)
                        mcts.fill_shared_input()

                    policy_logits, value = nnet.inference(shared_input)
                    for i, mcts in enumerate(mcts_list):
                        mcts.after_inference(policy_logits[i][0], value[i])

                # select action
                for i, mcts in enumerate(mcts_list):
                    if board_list[i].is_terminal():
                        continue
                    action = mcts.select_action(board_list[i].current_move, use_sampling=False)
                    board_list[i].take_action(action)
                    action_history[i].append(action)

            for i in range(num_mcts):
                if board_list[i].is_terminal() and done_list[i] == 0:
                    done_list[i] = 1
                    done_cnt += 1
            
            k+=1

        # win rate
        result = [0,0,0] # win, draw, lose
        for i in range(num_mcts):
            if board_list[i].winner == 1:
                result[2] += 1
            elif board_list[i].winner == -1:
                result[0] += 1
            else:
                result[1] += 1
        win_percentage = round(result[0]/(result[0]+result[2]) , 3)
        print(f'ai win/draw/lose : {result}, winning percentage : {win_percentage}')

        l = len(self.winrate_data[1])
        self.winrate_data[0].append(l+1)
        self.winrate_data[1].append(win_percentage)

        # save_replay
        for i in range(10):
            file_manager.save_replay(action_history[i], nnet.num_steps, self.config.num_simulations, board_list[i].winner,
                                     "random-play", f"mandarin-{nnet.num_steps}")

        
    def selfplay_game(self, nnet, replay_buffer: ReplayBuffer, file_manager: FileManager, save_replay=False):
        num_mcts = self.config.num_actors
        shared_input = np.zeros((num_mcts, 3, MAX_ROW, MAX_COL))  # neural net input

        # check all actors done
        done_list = [0]*num_mcts
        done_cnt = 0

        board_list = [Board() for _ in range(num_mcts)]
        board_history_list = [[] for _ in range(num_mcts)]
        pi_list = [[] for _ in range(num_mcts)]

        next_mcts_root = [None] * num_mcts
        k = 1

        selfplay_starttime = time.time()
        while done_cnt < num_mcts:
            print(f'{k}th move of {num_mcts} games.')
            k+=1

            mcts_list = []
            for i in range(num_mcts):
                mcts = MCTS(self.config, board_list[i], shared_input, i, next_mcts_root[i])
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
                    mcts.fill_shared_input()

                policy_logits, value = nnet.inference(shared_input)
                for i, mcts in enumerate(mcts_list):
                    mcts.after_inference(policy_logits[i][0], value[i])

            # select action
            for i, mcts in enumerate(mcts_list):
                if board_list[i].is_terminal():
                    continue

                board_history_list[i].append(board_list[i].get_board_state_to_evaluate())
                pi = self.get_search_statistics(mcts.root)
                pi_list[i].append(pi)

                action = mcts.select_action(board_list[i].current_move, use_sampling=True)
                board_list[i].take_action(action)

                if board_list[i].is_terminal() and done_list[i] == 0:
                    done_list[i] = 1
                    done_cnt += 1

                # keep next root node
                next_mcts_root[i] = mcts.root.children[action]
        
        # add to replay buffer
        for i in range(num_mcts):
            board_history = board_history_list[i]
            for j in range(len(board_history)):
                camp = Camp.Black if j%2 == 0 else Camp.White
                rw = board_list[i].get_terminal_value(camp)

                # 8 variation of board state
                augmented_bhl = self.augment_board_state(board_history_list[i][j])
                augmented_pil = self.augment_2d_list(pi_list[i][j])
                
                for bh, pi in zip(augmented_bhl, augmented_pil):
                    if j == 0:
                        continue # initial board state's all symmetries are same. use only one.
                    replay_buffer.append_reward_list(rw)
                    replay_buffer.append_board_history(bh)
                    replay_buffer.append_pi_list(pi)
        
        elapsed = time.time()-selfplay_starttime
        print(f'{num_mcts} games terminated. replay_buffer size is {len(replay_buffer.reward_list)}. elapsed time is {elapsed} seconds.')

    def augment_board_state(self, board_state):
        al = self.augment_2d_list(board_state[0])
        bl = self.augment_2d_list(board_state[1])
        cl = self.augment_2d_list(board_state[2])
        return [[a,b,c] for a,b,c in zip(al,bl,cl)]

    def augment_2d_list(self, lst):
        ret = [lst]
        for _ in range(3):
            right_rot = [list(a) for a in zip(*ret[-1][::-1])]
            ret.append(right_rot)
        
        ret.append(lst[::-1]) # flip ud
        for _ in range(3):
            right_rot = [list(a) for a in zip(*ret[-1][::-1])]
            ret.append(right_rot)
        
        return ret

    def train_network(self, replay_buffer: ReplayBuffer, file_manager: FileManager):
        nnet = self.nnet
        optimizer = torch.optim.Adam(nnet.parameters(), lr=1e-4)
        

        train_start = time.time()
        # prevent overfitting when replay_buffer small
        _training_step = min(self.config.training_steps, int(len(replay_buffer.board_history)*2 // self.config.batch_size)+1)
        
        print(f'training network.. step to train is {_training_step}')
        file_manager.save_replay_buffer(replay_buffer)
        for i in tqdm(range(_training_step)):
            if i % self.config.checkpoint_interval == 0:
                file_manager.save_checkpoint()
            
            batch = replay_buffer.sample_batch()
            self.update_weights(optimizer, nnet, batch, i%100==0)
            nnet.increase_num_steps()

        elapsed = time.time()-train_start
        print(f'finished training network. elapsed {elapsed} seconds')
        file_manager.save_checkpoint()
        file_manager.save_replay_buffer(replay_buffer)
    
    def loss_nll(self, outputs, targets):
        return -torch.sum(targets * outputs) / targets.size()[0]
    
    def update_weights(self, optimizer, nnet, batch, print_loss=False):
        mse_loss = nn.MSELoss()
        nll_loss = self.loss_nll
        
        image, target_policy, target_value = batch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        image = torch.Tensor(image).to(device)
        target_policy = torch.Tensor(target_policy).flatten(start_dim=1).to(device)
        target_value = torch.Tensor(target_value).to(device)

        policy_logits, value = nnet(image)
        loss1 = mse_loss(value, target_value)
        loss2 = nll_loss(policy_logits, target_policy) # model policy output is softmax. so log there to NLL Loss.
        loss = loss1+loss2

        if print_loss:
            print(loss1)
            print(loss2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.add_plot_data(nnet.num_steps, loss.cpu().detach().numpy())
    
    def add_plot_data(self, num_steps, loss):
        self.plot_data[0].append(num_steps)
        self.plot_data[1].append(loss)
        
    def save_plot_image(self, num_steps):
        plt.plot(self.plot_data[0], self.plot_data[1])
        plt.savefig(f'data/plots/{num_steps}.png')
        plt.close()
        
        plt.plot(self.winrate_data[0], self.winrate_data[1])
        plt.savefig(f'data/plots/winrate_{num_steps}.png')
        plt.close()

    def get_search_statistics(self, root):
        # pi history for training
        sum_visits = sum(child.visit_count for child in root.children.values())
        move_modality = [[[0]*1 for _ in range(MAX_COL)] for _ in range(MAX_ROW)]
        for action, child in root.children.items():
            if action[0] == -1:
                continue

            if sum_visits == 0:
                print('sum_visits=0', root)
                continue
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