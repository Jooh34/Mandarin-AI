import re
from collections import defaultdict
from copy import deepcopy

from core.board import Board
from core.types import Camp

from ui.constants import KEY_BLACK_PLAYER, KEY_WHITE_PLAYER, KEY_TOTAL_MOVE, KEY_MATCH_RESULT

class Gibo:
    '''
        match_info : name, han_formation, cho_formation, winner, move_counts ...
        actions : [move(str) ...]
        board_history : [_board ... ]
    '''
    
    def __init__(self, match_info, action_history, board_history):
        self.match_info = match_info
        self.action_history = action_history
        self.board_history = board_history

    @staticmethod
    def make_gibo_with_gib(file_path):
        match_info = defaultdict(str)
        action_history = []

        with open(file_path) as f:
            for line in f:
                m = re.match("\[(.*) \"(.*)\"\]", line)
                if m: # meta
                    try:
                        match_info[m.group(1)] = m.group(2)
                    except:
                        raise Exception(f'match_info parse error : wrong line in txt file : {line} \n in {file_path}')


                else:
                    action_match = re.findall("(.) (-?\d) (-?\d),", line)
                    if action_match: #actions
                        for camp, r, c in action_match:
                            action_history.append((int(r),int(c)))

        board_history = Gibo.make_board_history(match_info, action_history)
        return Gibo(match_info, action_history, board_history)

    @staticmethod
    def make_board_history(match_info, action_history):
        board = Board()

        board_history = []

        total_move = int(match_info[KEY_TOTAL_MOVE])
        board_history.append(deepcopy(board))

        for i, action in enumerate(action_history):
            board.take_action(action)
            board_history.append(deepcopy(board))

        if len(board_history) != total_move+1:
            raise Exception(f'board history : {len(board_history)} != total move+1 : {total_move+1}.')

        return board_history