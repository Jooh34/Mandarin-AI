import pygame

from enum import Enum
from ui.constants import BOARD_START_H, BOARD_START_W, ROW_GAP, COL_GAP, PIECE_HEIGHT, PIECE_WIDTH

RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
BLACK = (0,0,0)
WHITE = (255,255,255)

class MarkerType(Enum):
    SELECT = 0,
    NEXT_ACTION = 1,
    AI_THINK = 2,

class BoardMarker:
    def __init__(self, marker_type, r: int, c: int, prob = 0):
        self.marker_type = marker_type
        self.r = r
        self.c = c
        self.x, self.y = self.rowcol_to_pos(r,c)
        self.prob = round(prob,3)

    def draw(self, screen):
        if self.marker_type == MarkerType.NEXT_ACTION:
            pygame.draw.circle(screen, GREEN, (self.x, self.y), 7.5)
        
        elif self.marker_type == MarkerType.AI_THINK:
            pygame.draw.circle(screen, RED, (self.x, self.y), 15)
            content_font = pygame.font.SysFont("malgungothic", 11, False, False)
            prob_text = content_font.render(f"{self.prob}", True, WHITE)
            screen.blit(prob_text, (self.x-8, self.y-8))

    def rowcol_to_pos(self, row, col):
        return (BOARD_START_W + COL_GAP * col + PIECE_WIDTH, BOARD_START_H + ROW_GAP * row + PIECE_HEIGHT)
