from copy import deepcopy
import random
import torch
import numpy as np

from ai.mandarin_net import MandarinNet
from core.types import Camp
from core.board import Board
from core.move import Move, Action

class AIPlayer:
    def __init__(self, camp: Camp):
        self.camp = camp

        self.mandarin_net = MandarinNet()

    def turn(self, board: Board):
        new_board = deepcopy(board)
        if self.camp == Camp.HAN:
            new_board.rotate_and_reverse()
       
        possible_actions = Move.get_possible_actions(new_board._board)
        action_choice = self.choose_action(new_board, possible_actions)
        # random_action = random.choice(possible_actions)
        new_board.take_action(action_choice)
        if self.camp == Camp.HAN:
            new_board.rotate_and_reverse()

        return new_board

    def choose_action(self, board: Board, possible_actions: [Action]):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        print('ai choosing action..')
        p,v = self.mandarin_net.inference(board.get_board_state_to_evaluate())

        plist = []
        for action in possible_actions:
            i,j = action.prev
            move_type = action.move_type
            plist.append(p[move_type][i][j])

        prob = self.softmax(np.array(plist))
        index_max = np.argmax(prob)

        return possible_actions[index_max]

    def softmax(slef,x):
        exp_a = np.exp(x)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        
        return y