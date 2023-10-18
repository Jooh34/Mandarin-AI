import os
import torch
import pickle

from ai.mandarin_net import MandarinNet
from core.types import Formation

class FileManager(object):
    def __init__(self):
        self.nnet = None
        abs_path = os.path.dirname(__file__)
        self.checkpoint_folder = os.path.join(abs_path, '../../data/checkpoint')
        self.replay_folder = os.path.join(abs_path, '../../data/replay')
        self.replay_buffer_folder = os.path.join(abs_path, '../../data/replay_buffer')

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
            self.nnet = MandarinNet()
            return self.nnet

    def save_checkpoint(self):
        # change extension
        if not os.path.exists(self.checkpoint_folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(self.checkpoint_folder))
            os.mkdir(self.checkpoint_folder)

        filename = f'mandarin_{self.nnet.num_steps}' + ".pt"
        filepath = os.path.join(self.checkpoint_folder, filename)
        torch.save(self.nnet, filepath)

    def save_replay_buffer(self, replay_buffer):
        data = {
            'board_history' : list(replay_buffer.board_history),
            'pi_list' : list(replay_buffer.pi_list),
            'reward_list' : list(replay_buffer.reward_list)
        }

        filename = f'mandarin_{self.nnet.num_steps}' + ".pickle"
        filepath = os.path.join(self.replay_buffer_folder, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    def load_replay_buffer(self, num_steps, replay_buffer):
        print('loading replay buffer..')
        filename = f'mandarin_{num_steps}' + ".pickle"
        filepath = os.path.join(self.replay_buffer_folder, filename)
        if not os.path.exists(filepath):
            print(f'replay buffer {filepath} not exist. skip.')
            return

        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f'replay buffer successfully loaded. {len(data["pi_list"])} data loaded.')
        replay_buffer.load_from_pickle(data)

    def load_checkpoint(self, folder='checkpoint', filename='mandarin'):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))
        
        self.nnet = torch.load(filepath, map_location=device)

    def save_replay(self, action_history, num_steps, num_simulations, cho_formation: Formation, han_formation: Formation):
        file_name = f'replay_{num_steps}_sim{num_simulations}.gib'
        filepath = os.path.join(self.replay_folder, file_name)
        cnt = 1
        while os.path.exists(filepath):
            file_name = f'replay_{num_steps}_sim{num_simulations}_v{cnt}.gib'
            filepath = os.path.join(self.replay_folder, file_name)
            cnt+=1
            
        per_row = 6
        with open(filepath, "w", encoding='euc-kr') as f:
            f.write(f'[초차림 "{Formation.formatiopn_to_str(cho_formation)}"]\n')
            f.write(f'[한차림 "{Formation.formatiopn_to_str(han_formation)}"]\n')
            f.write(f'[총수 "{len(action_history)}"]\n')
            for row in range(len(action_history)//6 + 1):
                for j in range(6):
                    idx = row*per_row+j
                    if idx < len(action_history):
                        f.write(f'{idx+1}. {action_history[idx]} ')
                
                f.write('\n')