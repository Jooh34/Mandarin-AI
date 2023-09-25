from enum import IntEnum, Enum
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

class Camp(IntEnum):
    CHO = 1
    HAN = -1

MAX_ROW = 10
MAX_COL = 9