"""
``topology.py`` contains all the tools to transform a dataset
in an object that can be used by the main algorithm in tnn.py

Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
Written by Arnaud Pannatier <arnaud.pannatier@idiap.ch>
"""

import math

import numpy as np
import torch
import torch.nn.functional as F


def proper_padding(input, size):
    left = int(math.ceil((size - input.size(0)) / 2.0))
    right = int(math.floor((size - input.size(0)) / 2.0))

    return F.pad(input.view(1, 1, -1).float(), (left, right),
                 mode="replicate").squeeze().long()


class TrajectoryCreator:
    def __init__(self, X, y, trajectory_name_to_indices, max_depth=20):
        self.X = X
        self.y = y
        self.trajectory_name_to_indices = trajectory_name_to_indices
        self.max_depth = max_depth
        self.trajectories = self.create_trajectories()

        self.lines = np.array(
            [line for t in self.trajectories for line in t.components_list])

        self.TA = torch.cat([line.t[0].unsqueeze(0) for line in self.lines])
        self.TA, reorder_time = torch.sort(self.TA)
        self.TA = self.TA.contiguous()
        self.lines = self.lines[reorder_time.cpu().numpy()]

        self.TB = torch.cat([line.t[1].unsqueeze(0) for line in self.lines])
        self.A = torch.cat([line.start.unsqueeze(0) for line in self.lines])
        self.B = torch.cat([line.end.unsqueeze(0) for line in self.lines])
        self.U = torch.cat([line.u.unsqueeze(0) for line in self.lines])
        self.L = ((self.A - self.B)**2).sum(1)

        self.E = torch.cat(
            [line.max_euclidean_error().unsqueeze(0) for line in self.lines])
        self.create_line_idx_matrix()

    def create_trajectories(self):
        return [
            Trajectory(self.X, self.y, indices, name, self.max_depth)
            for name, indices in self.trajectory_name_to_indices.items()
        ]

    def create_line_idx_matrix(self):
        self.line_idx_matrix = torch.cat(
            [line.indices.unsqueeze(0) for line in self.lines])

    def create_dict_to_save(self):
        return {
            "line_idx_matrix": self.line_idx_matrix,
            "A": self.A.float(),
            "B": self.B.float(),
            "U": self.U.float(),
            "TA": self.TA.float(),
            "TB": self.TB.float(),
            "E": self.E.float(),
        }

    def save_trajectory_dict(self, filename):
        to_save = self.create_dict_to_save()

        torch.save(to_save, filename)

    def retrieve_index(self, indexes, batch):
        return self.line_idx_matrix[indexes]

    def get_matrices(self):
        return [self.A, self.B, self.U, self.TA, self.TB, self.E]


class MinimalTrajectories(TrajectoryCreator):
    def __init__(self, line_idx_matrix, A, B, U, TA, TB, E):
        self.line_idx_matrix = line_idx_matrix
        self.A = A
        self.B = B
        self.U = U
        self.TA = TA
        self.TB = TB
        self.E = E

    def add_trajectories(self, tc, shift):
        self.line_idx_matrix = torch.cat(
            (self.line_idx_matrix, tc.line_idx_matrix + shift), 0).long()
        self.A = torch.cat((self.A, tc.A), 0)
        self.B = torch.cat((self.B, tc.B), 0)
        self.U = torch.cat((self.U, tc.U), 0)
        self.TA = torch.cat((self.TA, tc.TA), 0)
        self.TB = torch.cat((self.TB, tc.TB), 0)
        self.E = torch.cat((self.E, tc.E), 0)

        print("{} trajectories added : total {}".format(
            tc.A.shape[0], self.A.shape[0]))


def load_trajectory_dict(filename, device):
    td_raw = torch.load(filename)
    td = {k: v.to(device) for k, v in td_raw.items()}

    return MinimalTrajectories(td["line_idx_matrix"], td["A"], td["B"],
                               td["U"], td["TA"], td["TB"], td["E"])


class Trajectory:
    def __init__(self, X, y, indices, trajectory_name, len_line):
        self.X = X
        self.y = y
        reorder_time = torch.argsort(self.X[indices, 3])
        self.indices = indices[reorder_time]
        self.trajectory_name = trajectory_name
        self.components_list = self.create_components_list(len_line=len_line)

    def create_components_list(self, len_line):
        equal_split = torch.split(self.indices, len_line)
        lines = [
            Line(self.X[idx[0], 0:3], self.X[idx[-1], 0:3], self.X, self.y,
                 idx) for idx in equal_split[:-1]
        ]

        if len(equal_split[-1]) > 1:
            idx = proper_padding(equal_split[-1], len_line)

            lines.append(
                Line(self.X[idx[0], 0:3], self.X[idx[-1], 0:3], self.X, self.y,
                     idx))

        return lines


class Line:
    def __init__(self, start, end, X, y, indices):
        self.start = start
        self.end = end
        self.X = X
        self.y = y
        self.indices = indices
        self.u = self.compute_unit_vector()
        self.t = torch.cat([
            X[self.indices[0], 3].unsqueeze(0), X[self.indices[-1],
                                                  3].unsqueeze(0)
        ])

    def compute_unit_vector(self):
        vector = self.end - self.start
        return vector / torch.norm(vector)

    def max_error(self):
        points = self.X[self.indices, :3]

        t = torch.clamp(((points - self.start) * self.u).sum(1), 0, 1)
        P_AB = t.unsqueeze(1) * (self.end - self.start) + self.start

        D = (P_AB - points)**2
        return torch.max(D, dim=0).values

    def max_euclidean_error(self):
        points = self.X[self.indices, :3]

        t = torch.clamp(((points - self.start) * self.u).sum(1), 0, 1)
        P_AB = t.unsqueeze(1) * (self.end - self.start) + self.start

        D = ((P_AB - points)**2).sum(1)
        return D.max()

    def plot(self, color="C1", lw=1):
        import matplotlib.pyplot as plt
        X = torch.cat([self.start.unsqueeze(0), self.end.unsqueeze(0)])
        plt.plot(X[:, 0], X[:, 1], color=color, lw=lw)
