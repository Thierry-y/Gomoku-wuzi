# coding: utf-8
from math import sqrt
from typing import Tuple, Iterable, Dict


class Node:
    """ Monte Carlo Tree Search (MCTS) Node """

    def __init__(self, prior_prob: float, c_puct: float = 5, parent=None):
        """
        Parameters
        ----------
        prior_prob: float
            Prior probability of the node `P(s, a)`

        c_puct: float
            Exploration constant

        parent: Node
            Parent node
        """
        self.Q = 0  # Average action value
        self.U = 0  # Upper confidence bound
        self.N = 0  # Visit count
        self.score = 0  # Node score
        self.P = prior_prob  # Prior probability
        self.c_puct = c_puct  # Exploration constant
        self.parent = parent  # Parent node
        self.children = {}  # type:Dict[int, Node]

    def select(self) -> tuple:
        """ Return the child node with the highest `score` and its corresponding action

        Returns
        -------
        action: int
            Action

        child: Node
            Child node
        """
        return max(self.children.items(), key=lambda item: item[1].get_score())

    def expand(self, action_probs: Iterable[Tuple[int, float]]):
        """ Expand the node by adding child nodes

        Parameters
        ----------
        action_probs: Iterable
            Each element is a tuple `(action, prior_prob)`, 
            which is used to create child nodes.
            The length of `action_probs` is equal to the number of available moves on the board.
        """
        for action, prior_prob in action_probs:
            self.children[action] = Node(prior_prob, self.c_puct, self)

    def __update(self, value: float):
        """ Update the visit count `N(s, a)` and the cumulative average reward `Q(s, a)`

        Parameters
        ----------
        value: float
            Value used to update the node's internal data
        """
        self.Q = (self.N * self.Q + value) / (self.N + 1)
        self.N += 1

    def backup(self, value: float):
        """ Backpropagate the value to update the tree """
        if self.parent:
            self.parent.backup(-value)

        self.__update(value)

    def get_score(self):
        """ Compute the node's score """
        self.U = self.c_puct * self.P * sqrt(self.parent.N) / (1 + self.N)
        self.score = self.U + self.Q
        return self.score

    def is_leaf_node(self):
        """ Check if the node is a leaf node """
        return len(self.children) == 0
