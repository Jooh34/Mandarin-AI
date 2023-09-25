from core.types import Formation, Piece, Camp

class Board:
    """
    Simple board class used for the game of Janggi. Contains and handles a single 
    10x9 two-dimensional list that contains either a Piece object or None.
    """

    def __init__(self, cho_formation: Formation, han_formation: Formation):
        self.board = [[0]*9 for _ in range(10)]
        self.cho_formation = cho_formation
        self.han_formation = han_formation

        self.init_bottom_formation(cho_formation, Camp.CHO)
        self.rotate()
        self.init_bottom_formation(han_formation, Camp.HAN)
        self.rotate()

    def get(self, r, c):
        return self.board[r][c]

    def set(self, r, c, v):
        self.board[r][c] = v
    
    def rotate(self):
        for _ in range(2):
            tuples = zip(*self.board[::-1])
            self.board = [list(elem) for elem in tuples]

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
            self.board[6][zol_col] = Piece.CHO_ZOL * camp
        
        self.board[7][1] = Piece.CHO_PO * camp
        self.board[7][7] = Piece.CHO_PO * camp

        self.board[9][0] = Piece.CHO_CHA * camp
        self.board[9][8] = Piece.CHO_CHA * camp

        self.board[9][3] = Piece.CHO_SA * camp
        self.board[9][5] = Piece.CHO_SA * camp

        self.board[8][4] = Piece.CHO_GOONG * camp

        if formation == Formation.MSMS:
            self.board[9][1] = Piece.CHO_MA * camp
            self.board[9][2] = Piece.CHO_SANG * camp
            self.board[9][6] = Piece.CHO_MA * camp
            self.board[9][7] = Piece.CHO_SANG * camp
        
        elif formation == Formation.MSSM:
            self.board[9][1] = Piece.CHO_MA * camp
            self.board[9][2] = Piece.CHO_SANG * camp
            self.board[9][6] = Piece.CHO_SANG * camp
            self.board[9][7] = Piece.CHO_MA * camp
        
        elif formation == Formation.SMMS:
            self.board[9][1] = Piece.CHO_SANG * camp
            self.board[9][2] = Piece.CHO_MA * camp
            self.board[9][6] = Piece.CHO_MA * camp
            self.board[9][7] = Piece.CHO_SANG * camp
        
        else: # formation == Formation.SMSM
            self.board[9][1] = Piece.CHO_SANG * camp
            self.board[9][2] = Piece.CHO_MA * camp
            self.board[9][6] = Piece.CHO_SANG * camp
            self.board[9][7] = Piece.CHO_MA * camp 