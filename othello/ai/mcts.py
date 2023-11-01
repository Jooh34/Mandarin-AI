import time
import math
import random
import tracemalloc
from copy import deepcopy
from collections import defaultdict
import numpy as np

from core.board import Board
from ai.config import AlphaZeroConfig
from typing import List

class Node:
    def __init__(self, prior: float):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0
        self.winner = None

        self.turn = -1
        self.children = {}

    def expanded(self):
        return len(self.children) > 0
    
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def is_terminal(self):
        return self.winner != None

    def __str__(self):
        s=[]
        s.append("value_sum: %s"%(self.value_sum))
        s.append("visit_count: %d"%(self.visit_count))
        s.append("children: %s"%(self.children.keys()))
        s.append("children visit count: %s"%([child.visit_count for child in self.children.values()]))
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
    def __init__(self, config: AlphaZeroConfig, board, shared_input, id, in_root_node=None):
        self.config = config
        self.timer = Timer()
        
        self.board = board
        self.scratch_game = board

        if in_root_node != None:
            self.root = in_root_node
        else:
            self.root = Node(0)
        self.root_turn = self.board.turn
        self.current_node = self.root
        self.search_path = []

        self.shared_input = shared_input
        self.id = id

        self.board_history = []
        self.pi_list = []

    def show_timer(self, reset=False):
        self.timer.show(reset)

    def fill_shared_input(self):
        board_np = self.board.get_board_state_to_evaluate()
        self.shared_input[self.id] = board_np

    def step(self, policy_logits, value):
        node = self.root
        scratch_game = deepcopy(self.board)
        search_path = [node]

        while node.expanded() and not node.is_terminal():
            action, node = self.select_child(node)
            scratch_game.take_action(action)
            node.winner = scratch_game.winner
            search_path.append(node)

        self.current_node = node
        self.search_path = search_path
        self.scratch_game = scratch_game

    def run_mcts(self, board: Board, nnet, num_simulations=1, add_exploration_noise=True):
        self.root = Node(0)
        self.evaluate(self.root, board, nnet)
        if add_exploration_noise:
            self.add_exploration_noise(self.root)

        for _ in range(num_simulations):
            self.mcts_one_step(board, nnet)

        return self.get_action_probability_value_list()
    
    def mcts_one_step(self, board: Board, nnet):
        node = self.root
        root_turn = self.root_turn
        scratch_game = deepcopy(board)
        search_path = [node]

        while node.expanded() and not node.is_terminal():
            action, node = self.select_child(node)
            scratch_game.take_action(action)
            node.winner = scratch_game.winner
            node.turn = scratch_game.turn
            search_path.append(node)

        if node.is_terminal():
            # print(f'predicted winner : {node.winner}, {node.turn}, root turn:{self.root.turn}=={root_turn}')
            if node.winner == node.turn:
                self.backpropagate(search_path, 1, node.turn)
            else:
                self.backpropagate(search_path, -1, node.turn)
                
        else:
            value = self.evaluate(node, scratch_game, nnet)
            self.backpropagate(search_path, value, node.turn)

    def get_action_probability_value_list(self):
        '''
        return action_probability_value_list : List[(prob, value, action)]
        '''
        root = self.root
        vc_vs_a_lst = [(child.visit_count, child.value_sum, action)
                        for action, child in root.children.items()]
        
        v_list = [v for v, _, _ in vc_vs_a_lst]
        v_sum = sum(v_list)
        ret = [(visit_count/v_sum, value_sum/v_sum, action) for visit_count, value_sum, action in vc_vs_a_lst]
        
        ret.sort(reverse=True)
        return ret

    def select_action(self, current_move, use_sampling=False):
        root = self.root
        visit_counts = [(child.visit_count, action)
                        for action, child in root.children.items()]

        # values = [(child.value_sum/(child.visit_count+1e-9), action) for action,child in root.children.items()]
        
        if current_move < self.config.num_sampling_moves and use_sampling:
            action = self.softmax_sample(visit_counts)
        else:
            _, action = max(visit_counts)
        return action
    
    def softmax_sample(self, visit_counts):
        v_list = [v for v, _ in visit_counts]
        v_sum = sum(v_list)
        p_list = [visit_count/v_sum for visit_count, _ in visit_counts]

        _, action = random.choices(visit_counts, weights=p_list, k=1)[0]
        return action

    def after_inference(self, policy_logits, value):
        # Expand the node.
        node = self.current_node

        node.turn = self.scratch_game.turn
        possible_actions = self.scratch_game.get_possible_actions(self.scratch_game.turn)
        policy = []

        for action in possible_actions:
            i,j = action
            p = policy_logits[i][j]
            policy.append(p)

        for i, p in enumerate(policy):
            node.children[possible_actions[i]] = Node(p)
        
        if node.is_terminal():
            if node.winner == node.turn:
                self.backpropagate(self.search_path, 1, node.turn)
            else:
                self.backpropagate(self.search_path, -1, node.turn)
                
        else:
            self.backpropagate(self.search_path, value, node.turn)
    
    # We use the neural network to obtain a value and policy prediction.
    def evaluate(self, node: Node, board: Board, nnet):
        input = np.array([board.get_board_state_to_evaluate()])
        policy_logits, value = nnet.inference(input)
        
        # Expand the node.
        node.turn = board.turn
        possible_actions = board.get_possible_actions(board.turn)
        policy = []

        for action in possible_actions:
            i,j = action
            p = policy_logits[0][0][i][j]
            policy.append(p)

        sum_p = 0
        for i, p in enumerate(policy):
            node.children[possible_actions[i]] = Node(p)
            sum_p += p
        
        # print(softmax_lst)
        
        return float(value)
    
    # At the end of a simulation, we propagate the evaluation all the way up the
    # tree to the root.
    def backpropagate(self, search_path: List[Node], value: float, turn):
        # print(f'root value_sum bef: {self.root.value_sum}')
        for node in search_path:
            # if node == self.root:
            #     print(f'backprop root : {value}, node.turn:{node.turn}, turn:{turn}')
            node.value_sum += (value if turn == node.turn else -value)
            node.visit_count += 1
        # print(f'root value_sum aft: {self.root.value_sum}')

    # At the start of each search, we add dirichlet noise to the prior of the root
    # to encourage the search to explore new actions.
    def add_exploration_noise(self, node: Node):
        actions = node.children.keys()
        noise = np.random.gamma(self.config.root_dirichlet_alpha, 1, len(actions))
        frac = self.config.root_exploration_fraction
        for a, n in zip(actions, noise):
            node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac

    def select_child(self, node: Node):
        _, action, child = max([(self.ucb_score(node, child), action, child)
                                for action, child in node.children.items()])
        
        # if node == self.root:
        #     for a, c in node.children.items():
        #         print(f'child_prior : {a}, prior: {c.prior} value: {round(float(c.value()),3)}, pvisit: {node.visit_count}, visit: {c.visit_count}, ucb:{self.ucb_score(node, c)}')
        
        return action, child
    def ucb_score(self, parent: Node, child: Node):
        pb_c = math.log((parent.visit_count + self.config.pb_c_base + 1) /
                        self.config.pb_c_base) + self.config.pb_c_init
        pb_c = self.config.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        value_score = -child.value()

        return prior_score + value_score