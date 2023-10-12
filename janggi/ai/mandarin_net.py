import torch
import torch.nn as nn

BOARD_H = 10
BOARD_W = 9
BOARD_C_IN = 2
BOARD_C_OUT = 59

class ProxyUniformNetwork(nn.Module):
    def __init__(self):
        pass

    def inference(self, _board):
        p = [[[0.5]*BOARD_C_OUT for _ in range(BOARD_W)] for _ in range(BOARD_H)]
        v = 0.5
        return p,v

class MandarinNet(nn.Module):
    def __init__(self):
        super().__init__()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # self.resnet = ResNet50()
        # self.value_head = ValueHead(512 * BottleNeck.expansion)
        # self.policy_head = PolicyHead(512 * BottleNeck.expansion)
        self.resnet = ResNet34()
        self.value_head = ValueHead(512)
        self.policy_head = PolicyHead(512)

        self.num_steps = 0

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
        # board : [9, 10, 2]
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        input = torch.tensor(_board, dtype=torch.float).to(device)
        input = input.unsqueeze(0) # [1,C,H,W]
        p,v = self(input)
        p = torch.reshape(p, (BOARD_C_OUT,BOARD_H,BOARD_W))
        p = p.cpu().detach().numpy()
        v = v.cpu().detach().numpy()

        return p,v


class PolicyHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # input : (BOARD_H, BOARD_W, 512)
        # output : (BOARD_H, BOARD_W, BOARD_C_OUT) # => move logit probabilities
        filter_size = 64
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, filter_size, 1, 1),
            nn.BatchNorm2d(filter_size),
            nn.ReLU(),
            nn.Conv2d(filter_size, BOARD_C_OUT, 1, 1),
            nn.Flatten() # to apply cross-entropy
        ).to(device)

    def forward(self, x):
        return self.model(x)

class ValueHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Use 1x1 filter CNN ? or GAP?

        # input : (BOARD_H, BOARD_W, 512)
        # output : FC - (BOARD_H * BOARD_W * filter_size)
        filter_size = 8

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, filter_size, 1, 1),
            nn.BatchNorm2d(filter_size),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(BOARD_H * BOARD_W * filter_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        ).to(device)
    
    def forward(self, x):
        return self.model(x)

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


        # input : (BOARD_H, BOARD_W, BOARD_C_IN)
        # output : (BOARD_H ,BOARD_W, 512)
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


        # input : (BOARD_H, BOARD_W, BOARD_C_IN)
        # output : (BOARD_H ,BOARD_W, 512)
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