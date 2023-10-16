import time
import math
import random
import numpy
import tracemalloc
from copy import deepcopy
from collections import defaultdict

from core.board import Board
from ai.config import AlphaZeroConfig
from typing import List

class Node:
    def __init__(self, prior: float):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0
        self.is_terminal = False

        self.turn = -1
        self.children = {}

    def expanded(self):
        return len(self.children) > 0
    
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def __str__(self):
        s=[]
        s.append("value_sum: %s"%(self.value_sum))
        s.append("visit_count: %d"%(self.visit_count))
        s.append("possible_actions: %s"%(self.children.keys()))
        return "%s: {%s}"%(self.__class__.__name__, ', '.join(s))

class Timer:
    def __init__(self, mem_trace=False):
        self.d_start = defaultdict(float)
        self.d_times = defaultdict(float)
        self.mem_trace = mem_trace
        if mem_trace:
            tracemalloc.start()

    def start(self, key):
        self.d_start[key] = time.time()

    def end(self, key):
        elapsed = time.time() - self.d_start[key]
        self.d_start[key] = 0
        self.d_times[key] += elapsed

    def reset_timer(self):
        self.d_start.clear()
        self.d_times.clear()

    def show(self, reset=False):
        for s,f in self.d_times.items():
            print(s, ": ", f)
        
        if reset:
            self.reset_timer()

        if self.mem_trace:
            print(tracemalloc.get_traced_memory())

class MCTS:
    def __init__(self, config: AlphaZeroConfig):
        self.config = config
        self.timer = Timer()

    def show_timer(self, reset=False):
        self.timer.show(reset)

    def run_mcts(self, board: Board, nnet):
        root = Node(0)
        self.evaluate(root, board, nnet)
        self.add_exploration_noise(root)

        for _ in range(self.config.num_simulations):
            node = root
            scratch_game = deepcopy(board)
            search_path = [node]

            while node.expanded() and not node.is_terminal:
                action, node = self.select_child(node)
                scratch_game.take_action_by_id(action)
                node.is_terminal = scratch_game.is_terminal()
                search_path.append(node)

            value = self.evaluate(node, scratch_game, nnet)
            self.backpropagate(search_path, value, scratch_game.turn)


        return self.select_action(board, root), root
    
    def select_action(self, board: Board, root: Node) -> str:
        visit_counts = [(child.visit_count, action)
                        for action, child in root.children.items()]
        
        # if len(board.history) < self.config.num_sampling_moves:
        #     _, action = softmax_sample(visit_counts)
        # else:
        _, action = max(visit_counts)
        return action

    # We use the neural network to obtain a value and policy prediction.
    def evaluate(self, node: Node, board: Board, nnet):
        policy_logits, value = nnet.inference(board.get_board_state_to_evaluate())
        
        # Expand the node.
        node.turn = board.turn
        possible_actions = board.get_possible_actions(board.turn)
        policy = []

        for action in possible_actions:
            i,j = action.prev
            move_type = action.move_type
            p = policy_logits[move_type][i][j]
            policy.append(math.exp(p))

        policy_sum = sum(policy)
        for i, p in enumerate(policy):
            node.children[possible_actions[i].get_unique_id()] = Node(p/policy_sum)
        
        return value
    
    # At the end of a simulation, we propagate the evaluation all the way up the
    # tree to the root.
    def backpropagate(self, search_path: List[Node], value: float, turn):
        for node in search_path:
            node.value_sum += value if node.turn == turn else (1 - value)
            node.visit_count += 1

    # At the start of each search, we add dirichlet noise to the prior of the root
    # to encourage the search to explore new actions.
    def add_exploration_noise(self, node: Node):
        actions = node.children.keys()
        noise = numpy.random.gamma(self.config.root_dirichlet_alpha, 1, len(actions))
        frac = self.config.root_exploration_fraction
        for a, n in zip(actions, noise):
            node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac

    # Select the child with the highest UCB score.
    def select_child(self, node: Node):
        _, action, child = max([(self.ucb_score(node, child), action, child)
                                for action, child in node.children.items()])
        
        return action, child
    def ucb_score(self, parent: Node, child: Node):
        pb_c = math.log((parent.visit_count + self.config.pb_c_base + 1) /
                        self.config.pb_c_base) + self.config.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        value_score = child.value()
        return prior_score + value_score