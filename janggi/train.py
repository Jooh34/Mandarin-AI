from core.board import Board
from core.types import Formation

from ai.mcts import MCTS

if __name__ == '__main__':
    initial_state = Board(Formation.MSMS, Formation.MSSM)
    searcher = MCTS(time_limit=1000)