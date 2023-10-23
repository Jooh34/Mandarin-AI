import logging
import pygame
import random
from pygame.locals import *
from copy import deepcopy

from ui.game_window import GameWindow
from ui.marker import BoardMarker, MarkerType
from core.board import Board
from core.types import Camp, MAX_COL, MAX_ROW
from core.move import Move
from ui.ai_player import AIPlayer

class GamePlayer:
    """play a single game using GameWindow."""

    def __init__(self):
        self.white_player = AIPlayer(Camp.White)
        self.board = Board()
        self.window = GameWindow(self.board)
        
        self.refresh_possible_action_marker()
        self.window.render()

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
            
            # Winner appeared
            if self.board.winner != None:
                self.window.set_winner()
                self.window.render()
                continue
            
            if self.board.turn == Camp.White:
                self.turn_ai()
                continue

            if event.type == MOUSEBUTTONDOWN and event.button == 1:
                px,py = pygame.mouse.get_pos()
                self.on_mousebutton_down(px,py)
    
    def on_take_action(self, action):
        self.board.take_action(action)

        self.window.switch_board(self.board)
        self.window.switch_markers([])
        self.window.render()
    
    def refresh_possible_action_marker(self):
        possible_actions = Move.get_possible_actions(self.board._board, self.board.turn)
        
        markers = []
        for action in possible_actions:
            markers.append(BoardMarker(MarkerType.NEXT_ACTION, action[0], action[1]))
            
        self.window.switch_markers(markers)

    def on_mousebutton_down(self,px,py):
        r,c = self.window.pos_to_rowcol(px,py)
        
        if r == -1 and c == -1: # only pass move
            possible_actions = Move.get_possible_actions(self.board._board, self.board.turn)
            if possible_actions[0][0] == -1:
                self.on_take_action((-1,-1))
                self.refresh_possible_action_marker()
                self.window.render()
                return

        if not (0 <= r < MAX_ROW and 0 <= c < MAX_COL):
            return

        possible_actions = Move.get_possible_actions(self.board._board, self.board.turn)
        for action in possible_actions:
            if action[0] == r and action[1] == c:
                self.on_take_action(action)
        
        self.refresh_possible_action_marker()
        self.window.render()

    def turn_user(self):
        pass

    def turn_ai(self):
        self.white_player.turn(self.board)

        self.window.switch_board(self.board)
        self.refresh_possible_action_marker()
        self.window.render()