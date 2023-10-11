import logging
import pygame
import random
from pygame.locals import *
from copy import deepcopy

from ui.game_window import GameWindow
from ui.marker import BoardMarker, MarkerType
from core.board import Board
from core.types import Formation, Piece, Camp
from core.move import Move
from ui.ai_player import AIPlayer

BOARD_H = 10
BOARD_W = 9
class GamePlayer:
    """play a single game using GameWindow."""
    current_move = 0

    def __init__(self):
        self.current_move = 0
        self.turn = 0 # 0 for cho, 1 for han

        self.han_player = AIPlayer(Camp.HAN)

        cho_form = random.randint(1,4)
        han_form = random.randint(1,4)
        self.board = Board(cho_form, han_form)
        self.possible_actions = Move.get_possible_actions(self.board._board)

        self.window = GameWindow(self.board)

        self.selected_piece = None

    def run(self):
        self.window.render()
        pygame.event.clear()
        while True:
            if self.turn == 1:
                self.turn_ai()
                continue

            event = pygame.event.wait()
            if event.type == QUIT:
                self.window.close()
                break

            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self.window.close()
                    break
            
            if event.type == MOUSEBUTTONDOWN and event.button == 1:
                px,py = pygame.mouse.get_pos()
                self.on_mousebutton_down(px,py)
    
    def on_take_action(self, action):
        self.board = self.board.take_action(action)

        self.current_move += 1
        self.turn = self.current_move % 2

        self.window.switch_board(self.current_move, self.board)
        self.window.switch_markers([])
        self.window.render()
        pass

    def on_mousebutton_down(self,px,py):
        r,c = self.window.pos_to_rowcol(px,py)
        if not (0 <= r <= BOARD_H and 0 <= c <= BOARD_W):
            return

        self.possible_actions = Move.get_possible_actions(self.board._board)

        if self.selected_piece:
            for action in self.possible_actions:
                if action.is_prev(self.selected_piece) and action.is_next((r,c)):
                    self.on_take_action(action)
                    
                    return

        if Piece.is_empty(self.board.get(r,c)) or Piece.is_enemy(self.board.get(r,c)):
            return

        # if first touch or not valid action.
        markers = []
        self.selected_piece = (r,c)
        markers.append(BoardMarker(MarkerType.SELECT, r, c))

        for action in self.possible_actions:
            if action.is_prev(self.selected_piece):
                markers.append(BoardMarker(MarkerType.NEXT_ACTION, action.next[0], action.next[1]))
        
        self.window.switch_markers(markers)
        self.window.render()

    def turn_ai(self):
        new_board = self.han_player.turn(self.board)

        self.board = new_board
        self.current_move += 1
        self.turn = self.current_move % 2

        self.window.switch_board(self.current_move, self.board)
        self.window.switch_markers([])
        self.window.render()