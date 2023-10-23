from copy import deepcopy
import random
import torch
import numpy as np

from core.types import Camp
from core.board import Board
from core.move import Move

class AIPlayer:
    def __init__(self, camp: Camp):
        self.camp = camp

    def turn(self, board: Board):
        possible_actions = Move.get_possible_actions(board._board, board.turn)
        random_action = random.choice(possible_actions)
        board.take_action(random_action)