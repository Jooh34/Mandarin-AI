from copy import deepcopy

from core.types import Camp, MAX_ROW, MAX_COL
from core.move import Move

class Board:
    """
    Simple board class used for the game of Othello. Contains and handles a single 
    8x8 two-dimensional list that contains -1 (White), 0 (Empty), 1 (Black).
    """

    def __init__(self):
        self._board = [[0]*MAX_COL for _ in range(MAX_ROW)]
        self.initialize_board()
        self.piece_count = {-1: 2, 1: 2}
        
        self.current_move = 0
        self.winner = None
        self.turn = Camp.Black

    def is_terminal(self):
        return self.winner != None

    def get(self, r, c):
        return self._board[r][c]

    def set(self, r, c, v):
        self._board[r][c] = v

    def initialize_board(self):
        self.set(3,3,Camp.White)
        self.set(4,4,Camp.White)
        self.set(3,4,Camp.Black)
        self.set(4,3,Camp.Black)

    def get_terminal_value(self, camp: Camp):
        if self.winner == camp:
            return 1
        elif self.winner == camp * (-1):
            return -1
        else:
            return 0 # draw

    def get_board_state_to_evaluate(self):
        turn_array = [[(1 if self.turn == 1 else 0)]*MAX_COL for _ in range(MAX_ROW)]
        black_array = [[(1 if self._board[i][j] == 1 else 0) for j in range(MAX_COL)] for i in range(MAX_ROW)]
        white_array = [[(1 if self._board[i][j] == -1 else 0) for j in range(MAX_COL)] for i in range(MAX_ROW)]
        # for b in black_array:
        #     print(b)
        # for w in white_array:
        #     print(w)
        # print(turn_array)
        return [black_array, white_array, turn_array]

    def get_possible_actions(self, turn: Camp):
        return Move.get_possible_actions(self._board, turn)

    def take_action(self, action):
        flip_count = Move.take_action(self._board, self.turn, action)
        self.piece_count[self.turn] += flip_count
        if action[0] != -1:
            self.piece_count[self.turn] += 1

        self.piece_count[-self.turn] -= flip_count

        self.current_move += 1
        self.turn *= -1

        # check terminal state
        self.check_terminal_state()

    def check_terminal_state(self):
        if self.piece_count[Camp.Black] == 0:
            self.winner = Camp.White
        
        if self.piece_count[Camp.White] == 0:
            self.winner = Camp.Black
        
        _sum = self.piece_count[Camp.Black] + self.piece_count[Camp.White]
        if _sum == 64:
            if self.piece_count[Camp.Black] < self.piece_count[Camp.White]:
                self.winner = Camp.White
            elif self.piece_count[Camp.Black] > self.piece_count[Camp.White]:
                self.winner = Camp.Black
            else:
                self.winner = 0
                

