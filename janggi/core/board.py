from core.types import Formation, Piece, Camp
from core.move import Move

from copy import deepcopy

class Board:
    """
    Simple board class used for the game of Janggi. Contains and handles a single 
    10x9 two-dimensional list that contains either a Piece object or None.
    """

    def __init__(self, cho_formation: Formation, han_formation: Formation):
        self._board = [[0]*9 for _ in range(10)]
        self.cho_formation = cho_formation
        self.han_formation = han_formation

        self.init_bottom_formation(cho_formation, Camp.CHO)
        self.rotate()
        self.init_bottom_formation(han_formation, Camp.HAN)
        self.rotate()

    def get(self, r, c):
        return self._board[r][c]

    def set(self, r, c, v):
        self._board[r][c] = v
    
    def rotate(self):
        for _ in range(2):
            tuples = zip(*self._board[::-1])
            self._board = [list(elem) for elem in tuples]

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

    def get_possible_actions(self):
        return Move.get_possible_actions(self._board)

    def take_action(self, action):
        new_state = deepcopy(self)
        pass