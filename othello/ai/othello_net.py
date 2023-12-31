import torch
import torch.nn as nn
import numpy as np

from core.types import MAX_COL, MAX_ROW
BOARD_C_IN = 3
BOARD_C_OUT = 1

NUM_RESIDUAL_LAYERS = 5
NUM_RESNET_CHANNEL = 256
HEAD_FC_SIZE = 512

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)

class OthelloNet(nn.Module):
    def __init__(self):
        super().__init__()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.resnet = ResNet()
        self.value_head = ValueHead(NUM_RESNET_CHANNEL)
        self.policy_head = PolicyHead(NUM_RESNET_CHANNEL)

        self.num_steps = 0

        # self.apply(initialize_weights)

    def increase_num_steps(self):
        self.num_steps += 1

    def set_num_steps(self, num_steps):
        self.num_steps = num_steps

    def get_num_steps(self):
        return self.num_steps

    def forward(self, x):
        x = self.resnet(x)
        v = self.value_head(x)
        p = self.policy_head(x)
        return p,v
    
    def inference(self, _board):
        # board : [3, MAX_ROW, MAX_COL]
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        input = torch.tensor(_board, dtype=torch.float).to(device)
        B = input.shape[0]
        # [1,H,W,C]
        p,v = self(input)
        # print(v)
        p = torch.exp(p)
        p = torch.reshape(p, (B,BOARD_C_OUT,MAX_ROW,MAX_COL))
        p = p.cpu().detach().numpy()
        v = v.cpu().detach().numpy()

        return p,v


class PolicyHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # input : (MAX_ROW, MAX_COL, in_channels)
        # output : (MAX_ROW, MAX_COL, BOARD_C_OUT) # => move logit probabilities
        filter_size = 32
        ret_dim = MAX_ROW*MAX_COL*BOARD_C_OUT
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, filter_size, 1, 1),
            nn.BatchNorm2d(filter_size),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(MAX_ROW*MAX_COL*filter_size, HEAD_FC_SIZE),
            nn.ReLU(),
            nn.Linear(HEAD_FC_SIZE, HEAD_FC_SIZE),
            nn.ReLU(),
            nn.Linear(HEAD_FC_SIZE, ret_dim),
            nn.LogSoftmax(dim=1),
        ).to(device)

    def forward(self, x):
        x = self.model(x)
        return x

class ValueHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Use 1x1 filter CNN ? or GAP?

        # input : (MAX_ROW, MAX_COL, in_channels)
        # output : 1 scalar (value of board)
        filter_size = 32

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, filter_size, 1, 1),
            nn.BatchNorm2d(filter_size),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(MAX_ROW * MAX_COL * filter_size, HEAD_FC_SIZE),
            nn.ReLU(),
            nn.Linear(HEAD_FC_SIZE, HEAD_FC_SIZE),
            nn.ReLU(),
            nn.Linear(HEAD_FC_SIZE, 1),
            nn.Tanh()
        ).to(device)
    
    def forward(self, x):
        return self.model(x)

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        layers = []

        num_blocks = [(NUM_RESIDUAL_LAYERS,NUM_RESNET_CHANNEL)]
        prev_channel = BOARD_C_IN
        for num_block,channel in num_blocks:
            for b in range(num_block):
                layers.append(BasicBlock(prev_channel, channel, 3, 1))
                
                prev_channel = channel


        # input : (MAX_ROW, MAX_COL, BOARD_C_IN)
        # output : (MAX_ROW ,MAX_COL, 64)
        self.model = nn.Sequential(*layers).to(device)

    def forward(self, x):
        x = self.model(x)
        return x
    
class ResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        layers = []

        num_blocks = [(3,64),(4,128),(6,256),(3,512)]
        prev_channel = BOARD_C_IN
        for num_block,channel in num_blocks:
            for b in range(num_block):
                layers.append(BasicBlock(prev_channel, channel, 3, 1))
                
                prev_channel = channel


        # input : (MAX_ROW, MAX_COL, BOARD_C_IN)
        # output : (MAX_ROW ,MAX_COL, 512)
        self.model = nn.Sequential(*layers).to(device)

    def forward(self, x):
        x = self.model(x)
        return x
    
class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        layers = []

        num_blocks = [(3,64),(4,128),(6,256),(3,512)]
        in_channels = BOARD_C_IN
        for num_block,channel in num_blocks:
            for b in range(num_block):
                layers.append(BottleNeck(in_channels, channel))
                
                in_channels = channel * BottleNeck.expansion


        # input : (MAX_ROW, MAX_COL, BOARD_C_IN)
        # output : (MAX_ROW ,MAX_COL, 512)
        self.model = nn.Sequential(*layers).to(device)

    def forward(self, x):
        x = self.model(x)
        return x

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding_size):
        super().__init__()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding_size),
            nn.BatchNorm2d(out_channels),
        ).to(device)

        self.shortcut = nn.Sequential()
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x

class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )
        
        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()

        if in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*BottleNeck.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels*BottleNeck.expansion)
            )
            
    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x