from copy import deepcopy
import random
import torch
import numpy as np

from core.types import Camp
from core.board import Board
from core.move import Move

from ai.othello_net import OthelloNet

class AIPlayer:
    def __init__(self, camp: Camp):
        self.camp = camp
        
        self.net = OthelloNet()

    def turn(self, board: Board):
        possible_actions = Move.get_possible_actions(board._board, board.turn)
        print(board)
        action_choice = self.choose_action(board, possible_actions)
        # random_action = random.choice(possible_actions)
        board.take_action(action_choice)

    def choose_action(self, board: Board, possible_actions):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        print('ai is choosing action..')
        p,v = self.net.inference(board.get_board_state_to_evaluate())

        plist = []
        for action in possible_actions:
            i,j = action
            print('action', action)
            if i == -1 and j == -1:
                plist.append(1)
            else:
                plist.append(p[0][i][j])

        prob = self.softmax(np.array(plist))
        index_max = np.argmax(prob)

        return possible_actions[index_max]

    def softmax(slef,x):
        exp_a = np.exp(x)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        
        return y