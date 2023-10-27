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
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.window.close()
                    break

                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        self.window.close()
                        break

                    if event.key == K_RETURN and self.board.turn == Camp.White:
                        self.action_ai()
                        self.window.switch_markers([])
                        continue

                if event.type == MOUSEBUTTONDOWN and event.button == 1 and self.board.turn == Camp.Black:
                    px,py = pygame.mouse.get_pos()
                    self.on_mousebutton_down(px,py)
                    continue

            self.window.render()
            # Winner appeared
            if self.board.winner != None:
                self.window.set_winner()
                self.window.render()
                continue 

            if self.board.turn == Camp.White:
                self.think_ai(self.board)
                self.window.render()
                continue

            self.refresh_possible_action_marker()

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
        self.window.render()

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

    def think_ai(self, board: Board):
        if self.white_player.mcts == None:
            self.white_player.initialize_mcts(board)
            return
        else:
            self.white_player.think_one_step(board)
        
        self.refresh_ai_marker()
    
    def refresh_ai_marker(self):
        apv_lst = self.white_player.action_probability_value_list
        markers = []
        for prob, value, action in apv_lst:
            markers.append(BoardMarker(MarkerType.AI_THINK, action[0], action[1], prob, value))
        
        self.window.switch_markers(markers)
        
    def action_ai(self):
        if self.white_player.mcts == None:
            raise("action_ai : mtcs is not initialized!")
        else:
            action = self.white_player.ai_choose_action()
            
            self.board.take_action(action) 
            self.window.switch_board(self.board)
            self.window.switch_markers([])
            self.window.render()