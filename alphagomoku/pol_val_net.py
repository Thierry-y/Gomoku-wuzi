# coding: utf-8
import torch
from torch import nn
from torch.nn import functional as F
from board import Board

class ConvBlock(nn.Module):
    """ Convolutional Block """

    def __init__(self, in_channels: int, out_channel: int, kernel_size, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channel,
                              kernel_size=kernel_size, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        return F.relu(self.batch_norm(self.conv(x)))


class ResidueBlock(nn.Module):
    """ Residual Block """

    def __init__(self, in_channels=128, out_channels=128):
        """
        Parameters
        ----------
        in_channels: int
            Number of input channels

        out_channels: int
            Number of output channels
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(num_features=out_channels)
        self.batch_norm2 = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        out = F.relu(self.batch_norm1(self.conv1(x)))
        out = self.batch_norm2(self.conv2(out))
        return F.relu(out + x)


class PolicyHead(nn.Module):
    """ Policy Head """

    def __init__(self, in_channels=128, board_len=9):
        """
        Parameters
        ----------
        in_channels: int
            Number of input channels

        board_len: int
            Board size
        """
        super().__init__()
        self.board_len = board_len
        self.in_channels = in_channels
        self.conv = ConvBlock(in_channels, 2, 1)
        self.fc = nn.Linear(2*board_len**2, board_len**2)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        return F.log_softmax(x, dim=1)


class ValueHead(nn.Module):
    """ Value Head """

    def __init__(self, in_channels=128, board_len=9):
        """
        Parameters
        ----------
        in_channels: int
            Number of input channels

        board_len: int
            Board size
        """
        super().__init__()
        self.in_channels = in_channels
        self.board_len = board_len
        self.conv = ConvBlock(in_channels, 1, kernel_size=1)
        self.fc = nn.Sequential(
            nn.Linear(board_len**2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        return x


class PolicyValueNet(nn.Module):
    """ Policy-Value Network """

    def __init__(self, board_len=9, n_feature_planes=6, is_use_gpu=True):
        """
        Parameters
        ----------
        board_len: int
            Board size

        n_feature_planes: int
            Number of input feature planes
        """
        super().__init__()
        self.board_len = board_len
        self.is_use_gpu = is_use_gpu
        self.n_feature_planes = n_feature_planes
        self.device = torch.device('cuda:0' if is_use_gpu else 'cpu')
        self.conv = ConvBlock(n_feature_planes, 128, 3, padding=1)
        self.residues = nn.Sequential(
            *[ResidueBlock(128, 128) for i in range(4)])
        self.policy_head = PolicyHead(128, board_len)
        self.value_head = ValueHead(128, board_len)

    def forward(self, x):
        """ Forward pass to output p_hat and V

        Parameters
        ----------
        x: Tensor of shape (N, C, H, W)
            Tensor representation of the board state

        Returns
        -------
        p_hat: Tensor of shape (N, board_len^2)
            Log prior probability vector

        value: Tensor of shape (N, 1)
            Estimated value of the current board state
        """
        x = self.conv(x)
        x = self.residues(x)
        p_hat = self.policy_head(x)
        value = self.value_head(x)
        return p_hat, value

    def predict(self, chess_board: Board):
        """ Get all available actions and their prior probabilities P(s, a),
            as well as the value of the board state.

        Parameters
        ----------
        chess_board: Board
            Chess board instance

        Returns
        -------
        probs: np.ndarray of shape (len(chess_board.available_actions), )
            Prior probabilities P(s, a) for all available actions

        value: float
            Estimated value of the current board state
        """
        feature_planes = chess_board.get_feature_planes().to(self.device)
        feature_planes.unsqueeze_(0)
        p_hat, value = self(feature_planes)

        # Convert log probabilities to probabilities
        p = torch.exp(p_hat).flatten()

        # Select only valid moves
        if self.is_use_gpu:
            p = p[chess_board.available_actions].cpu().detach().numpy()
        else:
            p = p[chess_board.available_actions].detach().numpy()

        return p, value[0].item()

    def set_device(self, is_use_gpu: bool):
        """ Set the device for neural network execution """
        self.is_use_gpu = is_use_gpu
        self.device = torch.device('cuda' if is_use_gpu else 'cpu')
