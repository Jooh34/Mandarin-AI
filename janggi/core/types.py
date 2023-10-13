from enum import IntEnum, Enum

import random

BOARD_H = 10
BOARD_W = 9
class Formation(Enum):
    """
        M: 마, S: 상
    """
    MSSM = 1
    SMMS = 2
    MSMS = 3
    SMSM = 4

    @staticmethod
    def str_to_formation(s: str):
        if s == '마상상마':
            return Formation.MSSM
        elif s == '상마마상':
            return Formation.SMMS
        elif s == '마상마상':
            return Formation.MSMS
        else: # s == '상마상마':
            return Formation.SMSM
        
    @staticmethod
    def formatiopn_to_str(f):
        if f == Formation.MSSM:
            return '마상상마'
        elif f == Formation.SMMS:
            return '상마마상'
        elif f == Formation.MSMS:
            return '마상마상'
        else: # f == Formation.SMSM:
            return '상마상마'
        
    @staticmethod
    def get_random_formation():
        i = random.randint(1,4)
        return Formation(i)
            

class Piece(IntEnum):
    CHO_ZOL = 1
    CHO_SANG = 2
    CHO_SA = 3
    CHO_MA = 4
    CHO_PO = 5
    CHO_CHA = 6
    CHO_GOONG = 7
    HAN_ZOL = -1
    HAN_SANG = -2
    HAN_SA = -3
    HAN_MA = -4
    HAN_PO = -5
    HAN_CHA = -6
    HAN_GOONG = -7

    @staticmethod
    def get_camp(piece):
        if piece > 0:
            return Camp.CHO
        else:
            return Camp.HAN

    @staticmethod
    def is_mine(piece):
        return piece > 0
    
    staticmethod
    def is_enemy(piece):
        return piece < 0
    
    @staticmethod
    def is_empty(piece):
        return piece == 0
    
    @staticmethod
    def is_po(piece):
        return piece == Piece.CHO_PO or piece == Piece.HAN_PO 

class Camp(IntEnum):
    CHO = 1
    HAN = -1

MAX_ROW = 10
MAX_COL = 9

class MoveType(IntEnum):
    """
    MoveType : Index for ML Model 9 x 10 x 59

    0~8	    위로 1~9칸
    9~17	아래로 1~9칸
    18~25	왼쪽으로 1~8칸
    26~33	오른쪽으로 1~8칸
    34~41	마- UP_LEFT 부터 시계방향 8가지
    42~49	상- UP_LEFT  시계방향 8가지
    50~57	대각선- UP_LEFT 부터 시계방향 8가지
    58	한수 쉼
    """
    MOVE_UP = 0,
    MOVE_DOWN = 9,
    MOVE_LEFT = 18,
    MOVE_RIGHT = 26,
    MA_UPLEFT = 34,
    SANG_UPLEFT = 42,
    DIAG_UPLEFT = 50,
    DIAG_UPRIGHT = 52,
    DIAG_DOWNLEFT = 54,
    DIAG_DOWNRIGHT = 56,
    PASS = 58

class Util:
    piece_to_kor_table = {1:'졸', 2:'상', 3:'사', 4:'마', 5:'포', 6:'차', 7:'장'}
    
    @staticmethod
    def piece_to_kor(piece):
        if Util.piece_to_kor_table.get(piece):
            return Util.piece_to_kor_table[piece]
        else:
            return '?'
        
    move_table = {
        MoveType.MA_UPLEFT: [[-1,0], [-1,-1]], # up-left
        MoveType.MA_UPLEFT+1: [[-1,0], [-1,1]], #up-right
        MoveType.MA_UPLEFT+2: [[0,1], [-1,1]], #right-up
        MoveType.MA_UPLEFT+3: [[0,1], [1,1]], # right-down
        MoveType.MA_UPLEFT+4: [[1,0], [1,1]], # down-right
        MoveType.MA_UPLEFT+5: [[1,0], [1,-1]], #down-left
        MoveType.MA_UPLEFT+6: [[0,-1], [1,-1]], #left-down
        MoveType.MA_UPLEFT+7: [[0,-1], [-1,-1]], #left-up

        MoveType.SANG_UPLEFT : [[-1,0], [-1,-1], [-1,-1]], # up-left
        MoveType.SANG_UPLEFT+1 : [[-1,0], [-1,1], [-1,1]], #up-right
        MoveType.SANG_UPLEFT+2: [[0,1], [-1,1], [-1,1]], #right-up
        MoveType.SANG_UPLEFT+3: [[0,1], [1,1], [1,1]], # right-down
        MoveType.SANG_UPLEFT+4: [[1,0], [1,1], [1,1]], # down-right
        MoveType.SANG_UPLEFT+5: [[1,0], [1,-1], [1,-1]], #down-left
        MoveType.SANG_UPLEFT+6: [[0,-1], [1,-1], [1,-1]], #left-down
        MoveType.SANG_UPLEFT+7: [[0,-1], [-1,-1], [-1,-1]], #left-up
    }

    @staticmethod
    def make_dir(move_type):
        return Util.move_table[move_type]