import pygame
import os, pkg_resources
from typing import List, Optional

from core.board import Board
from core.types import MAX_ROW, MAX_COL, Camp
from ui.constants import WIDTH, HEIGHT, BOARD_WIDTH, BOARD_HEIGHT, PIECE_HEIGHT, PIECE_WIDTH, BOARD_START_W, BOARD_START_H, ROW_GAP, COL_GAP

WHITE = (255,255,255)
BLACK = ( 0, 0, 0 )
IMG_PATH = "images/"
BOARD_FILENAME = "othello-board.png"


class Button:
    def __init__(self, display, img_in, x, y, width, height, on_click):
        mouse = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()
        if x < mouse[0] < x + width and y < mouse[1] < y + height and click[0]:
            on_click()

        display.blit(img_in, (x,y))

class GameWindow:
    """Class that renders board game."""

    def __init__(self, board):
        print('game window!')
        pygame.init()
        pygame.display.init()
        self.display = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Mandarin-AI Othello")

        self.board = board
        self.piece_imgs = {}
        self._init_board_image()
        self._init_piece_images()

        self.board_markers = []
        self.current_move = 0

        self.render()
    
    def render(self):
        self.display.fill(WHITE)
        if MAX_ROW == 8:
            self.display.blit(self.board_img, (0, 0))
        else:
            cropped_region = (0, 0, BOARD_WIDTH-150, BOARD_HEIGHT-150)
            self.display.blit(self.board_img, (0, 0), cropped_region)

        self._draw_pieces()
        # self._draw_match_info()

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
        camps = [-1, 1]
        for camp in camps:
            piece_path = self._get_image_path(camp)
            piece_file = pkg_resources.resource_filename( __name__, piece_path)
            piece_img = pygame.image.load(piece_file)
            self.piece_imgs[camp] = pygame.transform.scale(piece_img, (PIECE_WIDTH, PIECE_HEIGHT))
                
    def _get_image_path(self, camp):
        camp_name = "black" if camp == 1 else "white"
        filename = f"{camp_name}.png"
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

    def set_winner(self):
        header_font = pygame.font.SysFont("malgungothic", 20, True, False)

        result_str = ".."
        if self.board.winner != None:
            if self.board.winner == Camp.Black:
                result_str = "Black Win"
            elif self.board.winner == Camp.White:
                result_str = "White Win"
            else:
                result_str = "Draw"

        text_winner = header_font.render(f"Result : {result_str}", True, BLACK)
        self.display.blit(text_winner, (600, 100))
