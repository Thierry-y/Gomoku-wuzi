# coding: utf-8
from typing import Tuple, Union

import numpy as np

from board import Board
from node import Node
from pol_val_net import PolicyValueNet


class AlphaZeroMCTS:
    """ Monte Carlo Tree Search (MCTS) based on Policy-Value Network """

    def __init__(self, policy_value_net: PolicyValueNet, c_puct: float = 4, n_iters=1200, is_self_play=False) -> None:
        """
        Parameters
        ----------
        policy_value_net: PolicyValueNet
            Policy-Value Network

        c_puct: float
            Exploration constant

        n_iters: int
            Number of iterations

        is_self_play: bool
            Whether the MCTS is in self-play mode
        """
        self.c_puct = c_puct
        self.n_iters = n_iters
        self.is_self_play = is_self_play
        self.policy_value_net = policy_value_net
        self.root = Node(prior_prob=1, parent=None)

    def get_action(self, chess_board: Board) -> Union[Tuple[int, np.ndarray], int]:
        """ Return the next action based on the current game state

        Parameters
        ----------
        chess_board: Board
            Game board

        Returns
        -------
        action: int
            The best action for the current game state

        pi: `np.ndarray` of shape `(board_len^2, )`
            Probability distribution over the action space. 
            This is only returned when `is_self_play=True`.
        """
        for i in range(self.n_iters):
            # Copy the board
            board = chess_board.copy()

            # Search down the tree and update the board until reaching a leaf node
            node = self.root
            while not node.is_leaf_node():
                action, node = node.select()
                board.do_action(action)

            # Check if the game is over; if not, expand the leaf node
            is_over, winner = board.is_game_over()
            p, value = self.policy_value_net.predict(board)
            if not is_over:
                # Add Dirichlet noise for exploration
                if self.is_self_play:
                    p = 0.75 * p + 0.25 * np.random.dirichlet(0.03 * np.ones(len(p)))
                node.expand(zip(board.available_actions, p))
            elif winner is not None:
                value = 1 if winner == board.current_player else -1
            else:
                value = 0  # Draw

            # Backpropagation
            node.backup(-value)

        # Compute π (action probability distribution).
        # In self-play mode: during the first 30 moves, the temperature is 1; afterward, it approaches zero.
        T = 1 if self.is_self_play and len(chess_board.state) <= 30 else 1e-3
        visits = np.array([i.N for i in self.root.children.values()])
        pi_ = self.__getPi(visits, T)

        # Select action based on π and update root node
        actions = list(self.root.children.keys())
        action = int(np.random.choice(actions, p=pi_))

        if self.is_self_play:
            # Create π with shape (board_len^2)
            pi = np.zeros(chess_board.board_len**2)
            pi[actions] = pi_
            # Update the root node
            self.root = self.root.children[action]
            self.root.parent = None
            return action, pi
        else:
            self.reset_root()
            return action

    def __getPi(self, visits, T) -> np.ndarray:
        """ Compute π based on node visit counts """
        # Using visits**(1/T) / np.sum(visits**(1/T)) may cause numerical overflow,
        # so logarithmic scaling is applied.
        x = 1/T * np.log(visits + 1e-11)
        x = np.exp(x - x.max())
        pi = x / x.sum()
        return pi

    def reset_root(self):
        """ Reset the root node """
        self.root = Node(prior_prob=1, c_puct=self.c_puct, parent=None)

    def set_self_play(self, is_self_play: bool):
        """ Set self-play mode for the MCTS """
        self.is_self_play = is_self_play
