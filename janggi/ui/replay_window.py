import pygame
import os, pkg_resources
from typing import List, Optional


from core.board import Board
from core.gibo import Gibo
from core.types import MAX_ROW, MAX_COL, Piece

WIDTH, HEIGHT = 1000, 500
BOARD_WIDTH, BOARD_HEIGHT = 500, 500
PIECE_WIDTH, PIECE_HEIGHT = 50, 50
BOARD_START_W, BOARD_START_H = -24, -24
ROW_GAP, COL_GAP = 50, 56

WHITE = (255,255,255)
BLACK = ( 0, 0, 0 )
IMG_PATH = "images/"
BOARD_FILENAME = "board.png"

class ReplayWindow:
    """Class that renders board replay by gibo."""

    def __init__(self, match_info, board: Optional[Board] = None):
        print('replay window!')
        pygame.init()
        pygame.display.init()
        self.display = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Mandarin-AI")
        self.piece_imgs = {}

        self._init_board_image()
        self._init_piece_images()
        self._init_match_info(match_info)

        self.board = board
        self.current_move = 0

        self.render()
    
    def render(self):
        self.display.fill(WHITE)
        self.display.blit(self.board_img, (0, 0))
        self._draw_match_info()
        self._draw_pieces()

        pygame.event.pump()
        pygame.display.update()

    def close(self):
        pygame.quit()
    
    def switch_board(self, current_move, board):
        self.current_move = current_move
        self.board = board

    def _init_board_image(self):
        board_path = os.path.join(IMG_PATH, BOARD_FILENAME)
        board_file = pkg_resources.resource_filename(__name__, board_path)
        board_img = pygame.image.load(board_file)
        self.board_img = pygame.transform.scale( board_img, (BOARD_WIDTH, BOARD_HEIGHT))

    def _init_piece_images(self):
        for piece in Piece:
            piece_path = self._get_image_path(piece)
            piece_file = pkg_resources.resource_filename( __name__, piece_path)
            piece_img = pygame.image.load(piece_file)
            self.piece_imgs[piece] = pygame.transform.scale(piece_img, (PIECE_WIDTH, PIECE_HEIGHT))
                
    def _get_image_path(self, piece: Piece):
        filename = f"{piece.name.lower()}.png"
        return os.path.join(IMG_PATH, filename)
                
    def _init_match_info(self, match_info):
        self.cho_player = match_info[Gibo.KEY_CHO_PLAYER]
        self.han_player = match_info[Gibo.KEY_HAN_PLAYER]
        self.cho_formation = match_info[Gibo.KEY_CHO_FORMATION]
        self.han_formation = match_info[Gibo.KEY_HAN_FORMATION]
        self.total_move = match_info[Gibo.KEY_TOTAL_MOVE]
        self.match_result = match_info[Gibo.KEY_MATCH_RESULT]

    def _draw_pieces(self):
        # Draw pieces
        if self.board:
            for row in range(MAX_ROW):
                for col in range(MAX_COL):
                    piece = self.board.get(row,col)
                    if piece == 0:
                        continue
                    piece_img = self.piece_imgs[piece]
                    loc = (BOARD_START_W + COL_GAP * col + PIECE_WIDTH // 2, BOARD_START_H + ROW_GAP * row + PIECE_HEIGHT // 2)
                    self.display.blit(piece_img, loc)

    def _draw_match_info(self):
        header_font = pygame.font.SysFont("malgungothic", 20, True, False)
        content_font = pygame.font.SysFont("malgungothic", 11, False, False)

        text_players = header_font.render(f"{self.cho_player} (초) vs {self.han_player} (한)", True, BLACK)
        text_moves = header_font.render(f"{self.current_move} / {self.total_move}", True, BLACK)
        match_result = header_font.render(f"{self.match_result}", True, BLACK)

        self.display.blit(text_players, (600, 50))
        self.display.blit(match_result, (600, 100))
        self.display.blit(text_moves, (600, 150))



