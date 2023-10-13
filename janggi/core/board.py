from copy import deepcopy

from core.types import Formation, Piece, Camp
from core.move import Move, Action


BOARD_H = 10
BOARD_W = 9
BOARD_MOVE_MODAL = 59

class Board:
    """
    Simple board class used for the game of Janggi. Contains and handles a single 
    10x9 two-dimensional list that contains either a Piece object or None.
    """

    def __init__(self, cho_formation: Formation, han_formation: Formation):
        self._board = [[0]*BOARD_W for _ in range(BOARD_H)]
        self.cho_formation = cho_formation
        self.han_formation = han_formation
        self.winner = None

        self.current_move = 0
        self.turn = Camp.CHO
        self.pi_list = []

        self.init_bottom_formation(cho_formation, Camp.CHO)
        self.rotate()
        self.init_bottom_formation(han_formation, Camp.HAN)
        self.rotate()

    def is_terminal(self):
        return self.winner != None

    def get(self, r, c):
        return self._board[r][c]

    def set(self, r, c, v):
        self._board[r][c] = v
    
    def rotate(self):
        for _ in range(2):
            tuples = zip(*self._board[::-1])
            self._board = [list(elem) for elem in tuples]

    def rotate_and_reverse(self):
        self.rotate()
        for i in range(BOARD_H):
            for j in range(BOARD_W):
                self._board[i][j] = -self._board[i][j]

    def reverse_formation(self, formation: Formation):
        if formation == Formation.MSMS:
            return Formation.SMSM
        elif formation == Formation.SMSM:
            return Formation.MSMS
        elif formation == Formation.SMMS:
            return Formation.MSSM
        else: # formation == Formation.MSSM
            return Formation.SMMS

    def init_bottom_formation(self, formation: Formation, camp: Camp):
        if camp == Camp.HAN:
            formation = self.reverse_formation(formation)
            
        zol_cols = [0,2,4,6,8]
        for zol_col in zol_cols:
            self._board[6][zol_col] = Piece.CHO_ZOL * camp
        
        self._board[7][1] = Piece.CHO_PO * camp
        self._board[7][7] = Piece.CHO_PO * camp

        self._board[9][0] = Piece.CHO_CHA * camp
        self._board[9][8] = Piece.CHO_CHA * camp

        self._board[9][3] = Piece.CHO_SA * camp
        self._board[9][5] = Piece.CHO_SA * camp

        self._board[8][4] = Piece.CHO_GOONG * camp

        if formation == Formation.MSMS:
            self._board[9][1] = Piece.CHO_MA * camp
            self._board[9][2] = Piece.CHO_SANG * camp
            self._board[9][6] = Piece.CHO_MA * camp
            self._board[9][7] = Piece.CHO_SANG * camp
        
        elif formation == Formation.MSSM:
            self._board[9][1] = Piece.CHO_MA * camp
            self._board[9][2] = Piece.CHO_SANG * camp
            self._board[9][6] = Piece.CHO_SANG * camp
            self._board[9][7] = Piece.CHO_MA * camp
        
        elif formation == Formation.SMMS:
            self._board[9][1] = Piece.CHO_SANG * camp
            self._board[9][2] = Piece.CHO_MA * camp
            self._board[9][6] = Piece.CHO_MA * camp
            self._board[9][7] = Piece.CHO_SANG * camp
        
        else: # formation == Formation.SMSM
            self._board[9][1] = Piece.CHO_SANG * camp
            self._board[9][2] = Piece.CHO_MA * camp
            self._board[9][6] = Piece.CHO_SANG * camp
            self._board[9][7] = Piece.CHO_MA * camp

    def get_terminal_value(self, camp: Camp):
        if self.winner == camp:
            return 1
        elif self.winner == camp * (-1):
            return -1
        else:
            return 0 # draw

    def get_board_state_to_evaluate(self):
        if self.turn == Camp.HAN:
            self.rotate_and_reverse()
        
        turn_array = [[self.turn]*BOARD_W for _ in range(BOARD_H)]
        ret = [deepcopy(self._board), turn_array]
        if self.turn == Camp.HAN:
            self.rotate_and_reverse()

        return ret
    
    def make_target(self, state_index: int):
        turn = Camp.CHO if (state_index % 2) == 0 else Camp.HAN
        return (self.pi_list[state_index], self.get_terminal_value(turn))

    def get_possible_actions(self, turn: Camp):
        if turn == Camp.CHO:
            return Move.get_possible_actions(self._board)
        else: # HAN
            self.rotate_and_reverse()
            actions = Move.get_possible_actions(self._board)
            self.rotate_and_reverse()
            return actions
    
    def take_action(self, action: Action):
        turn_HAN = self.turn == Camp.HAN
        if turn_HAN:
            self.rotate_and_reverse()

        # judge winner
        ni, nj = action.next
        if self._board[ni][nj] == Piece.CHO_GOONG or self._board[ni][nj] == Piece.HAN_GOONG:
            self.set_winner(self.turn)

        # board state
        self.current_move += 1
        self.turn *= -1

        self.set(*action.prev, 0)
        self.set(*action.next, action.piece)

        if turn_HAN:
            self.rotate_and_reverse()

    def take_action_by_id(self, action_id: str):
        action = Action.init_by_id(action_id)
        self.take_action(action)
        
    def set_winner(self, camp: Camp):
        self.winner = camp

    def get_winner(self):
        return self.winner
