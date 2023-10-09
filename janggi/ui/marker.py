import pygame
from enum import Enum

RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

BOARD_START_W, BOARD_START_H = -24, -24
ROW_GAP, COL_GAP = 50, 56
PIECE_WIDTH, PIECE_HEIGHT = 50, 50

class MarkerType(Enum):
    SELECT = 0,
    NEXT_ACTION = 1,

class BoardMarker:
    def __init__(self, marker_type, r: int, c: int):
        self.marker_type = marker_type
        self.r = r
        self.c = c
        self.x, self.y = self.rowcol_to_pos(r,c)

    def draw(self, screen):
        if self.marker_type == MarkerType.SELECT:
            pygame.draw.circle(screen, RED, (self.x, self.y), 20, width=2)
        elif self.marker_type == MarkerType.NEXT_ACTION:
            pygame.draw.circle(screen, GREEN, (self.x, self.y), 7.5)

    def rowcol_to_pos(self, row, col):
        return (BOARD_START_W + COL_GAP * col + PIECE_WIDTH, BOARD_START_H + ROW_GAP * row + PIECE_HEIGHT)
