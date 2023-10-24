from copy import deepcopy
import random
import torch
import numpy as np

from core.types import Camp, MAX_COL, MAX_ROW
from core.board import Board

from ai.othello_net import OthelloNet
from ai.mcts import MCTS
from ai.config import AlphaZeroConfig

class AIPlayer:
    def __init__(self, camp: Camp):
        self.camp = camp
        self.nnet = OthelloNet()

        # to remove delay on first
        self.nnet.inference(np.zeros((1,3,MAX_ROW,MAX_COL)))

        self.mcts = None
        self.action_probabilities = None

    def initialize_mcts(self, board: Board):
        print('ai is choosing action..')
        self.mcts = MCTS(AlphaZeroConfig(), board, None, None)
        self.mcts.mcts_one_step(board, self.nnet)

    def think_one_step(self, board: Board):
        if not self.mcts:
            raise("mcts is not initialized.")
        
        self.mcts.mcts_one_step(board, self.nnet)
        self.action_probabilities = self.mcts.get_action_probabilities()
    
    def ai_choose_action(self):
        best_action = self.action_probabilities[0][1]

        self.mcts = None
        self.action_probabilities = None
       
        return best_action

    def softmax(slef,x):
        exp_a = np.exp(x)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        
        return y