# coding: utf-8
from typing import Tuple
from copy import deepcopy
from collections import OrderedDict

import torch
import numpy as np


class Board:
    """ Board class """

    EMPTY = -1
    WHITE = 0
    BLACK = 1

    def __init__(self, board_len=9, n_feature_planes=7):
        """
        Parameters
        ----------
        board_len: int
            Length of the board's side

        n_feature_planes: int
            Number of feature planes, must be even
        """
        self.board_len = board_len
        self.current_player = self.BLACK
        self.n_feature_planes = n_feature_planes
        self.available_actions = list(range(self.board_len**2))
        # Board state dictionary, where the key is the action, and the value is the current player
        self.state = OrderedDict()
        # Last move position
        self.previous_action = None

    def copy(self):
        """ Copy the board """
        return deepcopy(self)

    def clear_board(self):
        """ Clear the board """
        self.state.clear()
        self.previous_action = None
        self.current_player = self.BLACK
        self.available_actions = list(range(self.board_len**2))

    def do_action(self, action: int):
        """ Place a piece and update the board

        Parameters
        ----------
        action: int
            The position to place the piece, ranging from `[0, board_len^2 -1]`
        """
        self.previous_action = action
        self.available_actions.remove(action)
        self.state[action] = self.current_player
        self.current_player = self.WHITE + self.BLACK - self.current_player

    def do_action_(self, pos: tuple) -> bool:
        """ Place a piece and update the board, used only by the app

        Parameters
        ----------
        pos: Tuple[int, int]
            The position on the board to place the piece, ranging from `(0, 0) ~ (board_len-1, board_len-1)`

        Returns
        -------
        update_ok: bool
            Whether the move was successful
        """
        action = pos[0] * self.board_len + pos[1]
        if action in self.available_actions:
            self.do_action(action)
            return True
        return False

    def is_game_over(self) -> Tuple[bool, int]:
        """ Check if the game is over

        Returns
        -------
        is_over: bool
            Whether the game is over. It is `True` if there is a winner or a draw, otherwise `False`.

        winner: int
            The winner of the game, with the following possible values:
            * If the game has a winner, it will be `Board.BLACK` or `Board.WHITE`
            * If the game is not yet decided or ends in a draw, it will be `None`
        """
        # If fewer than 9 moves have been made, the game cannot be over
        if len(self.state) < 9:
            return False, None

        n = self.board_len
        act = self.previous_action
        player = self.state[act]
        row, col = act // n, act % n

        # Search directions
        directions = [[(0, -1),  (0, 1)],   # Horizontal search
                      [(-1, 0),  (1, 0)],   # Vertical search
                      [(-1, -1), (1, 1)],   # Main diagonal search
                      [(1, -1),  (-1, 1)]]  # Anti-diagonal search

        for i in range(4):
            count = 1
            for j in range(2):
                flag = True
                row_t, col_t = row, col
                while flag:
                    row_t = row_t + directions[i][j][0]
                    col_t = col_t + directions[i][j][1]
                    if 0 <= row_t < n and 0 <= col_t < n and self.state.get(row_t * n + col_t, self.EMPTY) == player:
                        # If the same color is encountered, increase count by 1
                        count += 1
                    else:
                        flag = False
            # Determine winner
            if count >= 5:
                return True, player

        # Draw
        if not self.available_actions:
            return True, None

        return False, None

    def get_feature_planes(self) -> torch.Tensor:
        """ Get the board state feature tensor, with shape `(n_feature_planes, board_len, board_len)`

        Returns
        -------
        feature_planes: Tensor of shape `(n_feature_planes, board_len, board_len)`
            Feature plane representation of the board
        """
        n = self.board_len
        feature_planes = torch.zeros((self.n_feature_planes, n**2))
        # The last plane represents the current player's color
        # feature_planes[-1] = self.current_player
        # Add historical moves
        if self.state:
            actions = np.array(list(self.state.keys()))[::-1]
            players = np.array(list(self.state.values()))[::-1]
            Xt = actions[players == self.current_player]
            Yt = actions[players != self.current_player]
            for i in range((self.n_feature_planes - 1) // 2):
                if i < len(Xt):
                    feature_planes[2 * i, Xt[i:]] = 1
                if i < len(Yt):
                    feature_planes[2 * i + 1, Yt[i:]] = 1

        return feature_planes.view(self.n_feature_planes, n, n)


