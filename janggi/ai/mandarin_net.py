import torch
import torch.nn as nn

BOARD_H = 10
BOARD_W = 9
BOARD_C_IN = 3
BOARD_C_OUT = 59

class MandarinNet(nn.Module):
    def __init__(self):
        super().__init__()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.resnet34 = ResNet34()
        self.valuenet = ValueNet()
        self.policynet = PolicyNet()

    def forward(self, x):
        x = self.resnet34(x)
        v = self.valuenet(x)
        p = self.policynet(x)
        return p,v

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # input : (BOARD_H, BOARD_W, 512)
        # output : (BOARD_H, BOARD_W, BOARD_C_OUT) # => move logit probabilities
        filter_size = 64
        self.model = nn.Sequential(
            nn.Conv2d(512, filter_size, 1, 1),
            nn.BatchNorm2d(filter_size),
            nn.ReLU(),
            nn.Conv2d(filter_size, BOARD_C_OUT, 1, 1),
        ).to(device)

    def forward(self, x):
        return self.model(x)

class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Use 1x1 filter CNN ? or GAP?

        # input : (BOARD_H, BOARD_W, 512)
        # output : FC - (BOARD_H * BOARD_W * filter_size)
        filter_size = 8

        self.model = nn.Sequential(
            nn.Conv2d(512, filter_size, 1, 1),
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
                if b == 0:
                    layers.append(ResidualLayer(prev_channel, channel, 3, 1, False))
                else:
                    layers.append(ResidualLayer(prev_channel, channel, 3, 1))
                
                prev_channel = channel


        # input : (BOARD_H, BOARD_W, BOARD_C_IN)
        # output : (BOARD_H ,BOARD_W, 512)
        self.model = nn.Sequential(*layers).to(device)

    def forward(self, x):
        x = self.model(x)
        return x

class ResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding_size, use_residual=True):
        super().__init__()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.use_residual = use_residual

        self.residual_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding_size),
            nn.BatchNorm2d(out_channels),
        ).to(device)

        self.relu = nn.ReLU()

    def forward(self, x):
        if self.use_residual:
            x = self.residual_block(x) + x
        else:
            x = self.residual_block(x)

        x = self.relu(x)
        return x