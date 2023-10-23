import logging
import pygame
from pygame.locals import *

from ui.replay_window import ReplayWindow
from core.gibo import Gibo
from core.board import Board


class ReplayViewer:
    """Display replay of a single game using ReplayWindow."""
    current_move = 0

    def __init__(self, filename: str):
        gibo = Gibo.make_gibo_with_gib(filename)
        self.gibo = gibo

        self.current_move = 0
        
        self.board = Board()
        self.window = ReplayWindow(gibo.match_info, self.board)

    def run(self):
        self.window.render()
        pygame.event.clear()
        while True:
            event = pygame.event.wait()
            if event.type == QUIT:
                self.window.close()
                break

            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self.window.close()
                    break

                elif event.key == K_RIGHT:
                    self._next()

                elif event.key == K_LEFT:
                    self._prev()
        
    def _next(self):
        if self.current_move < len(self.gibo.board_history)-1:
            self.current_move += 1
            board = self.gibo.board_history[self.current_move]

            self.window.switch_board(board)
            self.window.render()
        
        else:
            print('_next : this is last board history!')

    def _prev(self):
        if 0 < self.current_move:
            self.current_move -= 1
            board = self.gibo.board_history[self.current_move]
            self.window.switch_board(board)
            self.window.render()
        
        else:
            print('_prev : this is first board history!')