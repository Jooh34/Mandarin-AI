import os
import torch

from ai.mandarin_net import MandarinNet
from core.types import Formation

class FileManager(object):
    def __init__(self):
        self.nnet = None
        abs_path = os.path.dirname(__file__)
        self.checkpoint_folder = os.path.join(abs_path, 'checkpoint')
        self.replay_folder = os.path.join(abs_path, 'replay')

    def latest_network(self):
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

            print(f'num_step-{mx} checkpoint successfully loaded')

            return self.nnet
        else:
            # return ProxyUniformNetwork()  # policy -> uniform, value -> 0.5
            self.nnet = MandarinNet()
            return self.nnet

    def save_checkpoint(self):
        filename = f'mandarin_{self.nnet.num_steps}'

        # change extension
        if not os.path.exists(self.checkpoint_folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(self.checkpoint_folder))
            os.mkdir(self.checkpoint_folder)

        filename = f'mandarin_{self.nnet.num_steps}' + ".pt"
        filepath = os.path.join(self.checkpoint_folder, filename)
        torch.save(self.nnet, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='mandarin'):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))
        
        self.nnet = torch.load(filepath, map_location=device)

    def save_replay(self, action_history, num_steps, num_simulations, cho_formation: Formation, han_formation: Formation):
        file_name = f'replay_{num_steps}_sim{num_simulations}.gib'
        filepath = os.path.join(self.replay_folder, file_name)
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