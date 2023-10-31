from copy import deepcopy
import random
import torch
import numpy as np

from core.types import Camp, MAX_COL, MAX_ROW
from core.board import Board

from ai.othello_net import OthelloNet
from ai.mcts import MCTS
from ai.config import AlphaZeroConfig
from ai.file_manager import FileManager
from ai.trainer import ReplayBuffer

class AIPlayer:
    def __init__(self, camp: Camp):
        self.camp = camp
        
        self.config = AlphaZeroConfig()
        file_manager = FileManager()
        self.nnet = file_manager.latest_network()

        # to remove delay on first
        self.nnet.inference(np.zeros((1,3,MAX_ROW,MAX_COL)))

        self.mcts = None
        self.action_probability_value_list = None

    def initialize_mcts(self, board: Board):
        print('ai is choosing action..')
        self.mcts = MCTS(self.config, board, None, None)
        self.mcts.mcts_one_step(board, self.nnet)

    def think_one_step(self, board: Board):
        if not self.mcts:
            raise("mcts is not initialized.")
        
        self.mcts.mcts_one_step(board, self.nnet)
        self.action_probability_value_list = self.mcts.get_action_probability_value_list()
    
    def ai_choose_action(self):
        best_action = self.action_probability_value_list[0][2]

        self.mcts = None
        self.action_probability_value_list = None
       
        return best_action