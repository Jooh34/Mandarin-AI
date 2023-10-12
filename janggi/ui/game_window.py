import pygame
import os, pkg_resources
from typing import List, Optional

from core.board import Board
from core.types import MAX_ROW, MAX_COL, Piece, Camp

WIDTH, HEIGHT = 1000, 500
BOARD_WIDTH, BOARD_HEIGHT = 500, 500
PIECE_WIDTH, PIECE_HEIGHT = 50, 50
BOARD_START_W, BOARD_START_H = -24, -24
ROW_GAP, COL_GAP = 50, 56

WHITE = (255,255,255)
BLACK = ( 0, 0, 0 )
IMG_PATH = "images/"
BOARD_FILENAME = "board.png"

class GameWindow:
    """Class that renders board game."""

    def __init__(self, board):
        print('game window!')
        pygame.init()
        pygame.display.init()
        self.display = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Mandarin-AI")

        self.board = board
        self.piece_imgs = {}

        self._init_board_image()
        self._init_piece_images()
        self.board_markers = []

        self.current_move = 0

        self.render()
    
    def render(self):
        self.display.fill(WHITE)
        self.display.blit(self.board_img, (0, 0))
        self._draw_pieces()
        self._draw_match_info()

        # Draw markers
        for marker in self.board_markers:
            marker.draw(self.display)

        pygame.event.pump()
        pygame.display.update()

    def close(self):
        pygame.quit()
    
    def switch_board(self, board: Board):
        self.board = board
        self.current_move = board.current_move

    def switch_markers(self, markers):
        self.board_markers = markers
        
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

    def _draw_pieces(self):
        # Draw pieces
        if self.board:
            for row in range(MAX_ROW):
                for col in range(MAX_COL):
                    piece = self.board.get(row,col)
                    if piece == 0:
                        continue
                    piece_img = self.piece_imgs[piece]
                    loc = self.rowcol_to_pos(row,col)
                    self.display.blit(piece_img, loc)

    def rowcol_to_pos(self,row,col):
        return (BOARD_START_W + COL_GAP * col + PIECE_WIDTH // 2, BOARD_START_H + ROW_GAP * row + PIECE_HEIGHT // 2)
    
    def pos_to_rowcol(self,px,py):
        return ((py-BOARD_START_H-PIECE_HEIGHT//2)//ROW_GAP, (px-BOARD_START_W-PIECE_WIDTH//2)//COL_GAP)

    def _draw_match_info(self):
        header_font = pygame.font.SysFont("malgungothic", 20, True, False)
        content_font = pygame.font.SysFont("malgungothic", 11, False, False)

        text_moves = header_font.render(f"Move : {self.board.current_move}", True, BLACK)
        if self.board.winner != None:
            winner_kor = '초' if self.board.winner == Camp.CHO else '한'
            text_winner = header_font.render(f"Winner : {winner_kor}", True, BLACK)
            self.display.blit(text_winner, (600, 100))

        self.display.blit(text_moves, (600, 150))