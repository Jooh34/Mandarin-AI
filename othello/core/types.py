from enum import IntEnum, Enum

MAX_ROW = 6
MAX_COL = 6

WIDTH, HEIGHT = 1000, 500
BOARD_WIDTH, BOARD_HEIGHT = 500, 500
PIECE_WIDTH, PIECE_HEIGHT = 50, 50
BOARD_START_W, BOARD_START_H = 25, 25
ROW_GAP, COL_GAP = 50, 50

class Camp(IntEnum):
    Black = 1
    White = -1