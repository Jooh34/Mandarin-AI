import os
import torch
import pickle

from ai.othello_net import OthelloNet
from ui.constants import KEY_BLACK_PLAYER, KEY_WHITE_PLAYER, KEY_TOTAL_MOVE, KEY_MATCH_RESULT

class FileManager(object):
    def __init__(self):
        self.nnet = None
        abs_path = os.path.dirname(__file__)
        self.checkpoint_folder = os.path.join(abs_path, '../data/checkpoint')
        self.replay_folder = os.path.join(abs_path, '../data/replay')
        self.replay_buffer_folder = os.path.join(abs_path, '../data/replay_buffer')

    def latest_network(self, replay_buffer = None):
        lst = os.listdir(self.checkpoint_folder)
        candidate = []
        for file_name in lst:
            ext = file_name.split('.')[-1]
            if ext == 'pt':
                candidate.append(file_name)

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
            if replay_buffer != None:
                self.load_replay_buffer(self.nnet.num_steps, replay_buffer)

            print(f'num_step-{mx} checkpoint successfully loaded')

            return self.nnet
        else:
            # return ProxyUniformNetwork()  # policy -> uniform, value -> 0.5
            self.nnet = OthelloNet()
            if replay_buffer != None:
                self.load_replay_buffer(self.nnet.num_steps, replay_buffer)
            return self.nnet

    def save_checkpoint(self):
        # change extension
        if not os.path.exists(self.checkpoint_folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(self.checkpoint_folder))
            os.mkdir(self.checkpoint_folder)

        filename = f'mandarin_othello_{self.nnet.num_steps}' + ".pt"
        filepath = os.path.join(self.checkpoint_folder, filename)
        torch.save(self.nnet, filepath)

    def save_replay_buffer(self, replay_buffer):
        data = {
            'board_history' : list(replay_buffer.board_history),
            'pi_list' : list(replay_buffer.pi_list),
            'reward_list' : list(replay_buffer.reward_list)
        }

        filename = f'mandarin_othello_{self.nnet.num_steps}' + ".pickle"
        filepath = os.path.join(self.replay_buffer_folder, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    def load_replay_buffer(self, num_steps, replay_buffer):
        print('loading replay buffer..')
        filename = f'mandarin_othello_{num_steps}' + ".pickle"
        filepath = os.path.join(self.replay_buffer_folder, filename)
        if not os.path.exists(filepath):
            print(f'replay buffer {filepath} not exist. skip.')
            return

        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f'replay buffer successfully loaded. {len(data["pi_list"])} data loaded.')
        replay_buffer.load_from_pickle(data)

        # black_win = 0
        # white_win = 0
        # for i in range(100000):
        #     bh = replay_buffer.board_history[i]
        #     rw = replay_buffer.reward_list[i]
        #     pi = replay_buffer.pi_list[i]

        #     if bh[2][0][0] == 0 and rw == 1:
        #         black_win+=1
        #     if bh[2][0][0] == 1 and rw == 1:
        #         white_win+=1

        #     print(f'turn : {bh[2][0][0]}, reward : {rw}')
        #     # for row in pi:
        #     #     print(row)
        # print(f'black win : {black_win}, white win : {white_win}')
        # raise(Exception("asdasd"))

    def load_checkpoint(self, folder, filename):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))
        
        self.nnet = torch.load(filepath, map_location=device)

    def save_replay(self, action_history, num_steps, num_simulations, winner, black_player='', white_player=''):
        file_name = f'replay_{num_steps}_sim{num_simulations}.txt'
        filepath = os.path.join(self.replay_folder, file_name)
        cnt = 1
        while os.path.exists(filepath):
            file_name = f'replay_{num_steps}_sim{num_simulations}_v{cnt}.txt'
            filepath = os.path.join(self.replay_folder, file_name)
            cnt+=1
            
        per_row = 6
        with open(filepath, "w", encoding='euc-kr') as f:
            f.write(f'[{KEY_BLACK_PLAYER} {black_player}]\n')
            f.write(f'[{KEY_WHITE_PLAYER} {white_player}]\n')
            f.write(f'[{KEY_MATCH_RESULT} "{winner}"]\n')
            f.write(f'[{KEY_TOTAL_MOVE} "{len(action_history)}"]\n')
            for row in range(len(action_history)//6 + 1):
                for j in range(6):
                    idx = row*per_row+j
                    if idx < len(action_history):
                        s = 'B' if idx % 2 == 0 else 'W'
                        f.write(f'{s} {action_history[idx][0]} {action_history[idx][1]},')
                
                f.write('\n')