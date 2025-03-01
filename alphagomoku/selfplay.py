# coding:utf-8
from collections import deque, namedtuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

SelfPlayData = namedtuple(
    'SelfPlayData', ['pi_list', 'z_list', 'feature_planes_list'])


class SelfPlayDataSet(Dataset):
    """ Self-play dataset class, where each sample is a tuple `(feature_planes, pi, z)` """

    def __init__(self, board_len=9):
        super().__init__()
        self.__data_deque = deque(maxlen=10000)
        self.board_len = board_len

    def __len__(self):
        return len(self.__data_deque)

    def __getitem__(self, index):
        return self.__data_deque[index]

    def clear(self):
        """ Clear the dataset """
        self.__data_deque.clear()

    def append(self, self_play_data: SelfPlayData):
        """ Insert data into the dataset """
        n = self.board_len
        z_list = Tensor(self_play_data.z_list)
        pi_list = self_play_data.pi_list
        feature_planes_list = self_play_data.feature_planes_list
        # Use rotation and mirroring to augment the dataset
        for z, pi, feature_planes in zip(z_list, pi_list, feature_planes_list):
            for i in range(4):
                # Rotate counterclockwise by i * 90°
                rot_features = torch.rot90(Tensor(feature_planes), i, (1, 2))
                rot_pi = torch.rot90(Tensor(pi.reshape(n, n)), i)
                self.__data_deque.append(
                    (rot_features, rot_pi.flatten(), z))

                # Horizontally flip the rotated arrays
                flip_features = torch.flip(rot_features, [2])
                flip_pi = torch.fliplr(rot_pi)
                self.__data_deque.append(
                    (flip_features, flip_pi.flatten(), z))
