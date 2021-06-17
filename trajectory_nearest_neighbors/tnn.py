"""
``tnn.py`` contains all the core function of the package.
The main function query which retrive the kNN of the batch
And a function distance_to_segments
The rest is some helpers functions

Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
Written by Arnaud Pannatier <arnaud.pannatier@idiap.ch>

This file is part of trajectory_nearest_neighbors

trajectory_nearest_neighbors is free software: you can redistribute it and/or
modify it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

trajectory_nearest_neighbors is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import torch


def distance_to_segments(trajs, batch, time_window, s=None):
    b = batch[:, :3].unsqueeze(1)

    if s is None:
        s = torch.ones(4, device=b.device)

    if len(s.shape) == 1:
        s = s.unsqueeze(0)

    A, B = trajs.A, trajs.B
    TA, TB = trajs.TA, trajs.TB
    E, U = trajs.E.unsqueeze(0), trajs.U
    t = torch.clamp((-(A.unsqueeze(0) - b) * U).sum(2), 0, 1)
    error = s[:, :3].max(1).values.unsqueeze(1).matmul(E)

    P_AB = t.unsqueeze(2) * (B - A) + A

    D = ((P_AB - b)**2) * s[:, :3].unsqueeze(1)
    D = D.sum(2)
    t = (batch[:, 3] - time_window).unsqueeze(1)
    D += torch.clamp((TB - t), min=0)**2 * s[:, 3].unsqueeze(1)
    D[TA > t] = float("inf")

    dist_with_error = torch.clamp(D - error, min=0)
    del b
    del t
    del P_AB
    del D
    del error

    return torch.sort(dist_with_error)


def distance_to_segments_with_lim(trajs,
                                  batch,
                                  time_window,
                                  s=None,
                                  limit_bound_idx=None):
    b = batch[:, :3].unsqueeze(1)

    if s is None:
        s = torch.ones(4, device=b.device)

    if len(s.shape) == 1:
        s = s.unsqueeze(0)

    mi = get_indices_between_boundaries(batch[:, 3], trajs.TA, limit_bound_idx)
    A, B = trajs.A[mi], trajs.B[mi]
    TA, TB = trajs.TA[mi], trajs.TB[mi]
    E, U = trajs.E[mi], trajs.U[mi]
    t = torch.clamp((-(A - b) * U).sum(2), 0, 1)
    error = E * s[:, :3].max(1).values.unsqueeze(1)

    P_AB = t.unsqueeze(2) * (B - A) + A
    D = ((P_AB - b)**2) * s[:, :3].unsqueeze(1)
    D = D.sum(2)
    t = (batch[:, 3] - time_window).unsqueeze(1)
    D += torch.clamp((TB - t), min=0)**2 * s[:, 3].unsqueeze(1)
    D[TA > t] = float("inf")

    dist_with_error = torch.clamp(D - error, min=0)
    del b
    del t
    del P_AB
    del D
    del error

    D, idx = torch.sort(dist_with_error)
    return D, mi.gather(1, idx)


def query(tc,
          X,
          batch_o,
          time_window,
          n_lines=4,
          k=10,
          s=None,
          verbose=False,
          limit_bound_idx=None):
    batch = batch_o.detach().clone()
    device = batch.device
    if s is None:
        sigmas = torch.ones(4, device=device)
    else:
        sigmas = s.detach().clone()

    # [batch size, L]
    if limit_bound_idx is None:
        lines_d, lines_idx = distance_to_segments(tc, batch, time_window,
                                                  sigmas)
    else:
        lines_d, lines_idx = distance_to_segments_with_lim(
            tc, batch, time_window, sigmas, limit_bound_idx)

    # Setting up the different usefull tensors
    next_index = torch.tensor(0, device=device).int()
    neighbors_dist, neighbors_index, converged_index, ordering = setup_tensors(
        batch, k)
    i = torch.tensor(0, device=device).int()
    c = torch.tensor(0, device=device).int()

    if len(sigmas.shape) == 1 or sigmas.shape[0] == 1:
        sigmas = sigmas.repeat(batch.shape[0], 1)

    while converged_index != 0 and next_index < lines_idx.shape[1] - 1:
        i += 1

        # Fetch F lines of P point
        # [searched batch, n_lines, points_per_line]
        fi = next_index+n_lines if next_index + \
            n_lines < lines_idx.shape[1] else lines_idx.shape[1]

        points_index = tc.retrieve_index(
            lines_idx[:converged_index, next_index:fi],
            batch[:converged_index])
        points_index = points_index.view(points_index.shape[0], -1)
        c += converged_index * points_index.shape[1]
        points = X[points_index]

        # [searched batch, (n_lines * points_per_line), 4]
        next_index += n_lines
        if next_index >= lines_idx.shape[1]:
            next_index = lines_idx.shape[1] - 1
        diff = (points - batch[:converged_index, :4].unsqueeze(1))**2
        unsorted_dists = (diff * sigmas[:converged_index].unsqueeze(1)).sum(2)

        mask = points[:, :, 3] >= (batch[:converged_index, 3] -
                                   time_window).unsqueeze(1)

        unsorted_dists[mask] = float("inf")

        # [searched batch, k+(n_lines * points_per_lines)]
        unsorted_dists = torch.cat(
            (neighbors_dist[:converged_index], unsorted_dists), 1)
        points_index = torch.cat(
            (neighbors_index[:converged_index], points_index), 1)

        # [not converged batch size, n_lines, points_per_line ]
        nn_d, nn_idx = unsorted_dists.topk(k, largest=False, sorted=True)

        neighbors_index[:converged_index] = points_index.gather(1, nn_idx)
        neighbors_dist[:converged_index] = nn_d

        next_dists = lines_d[:converged_index, next_index]
        newly_converged = next_dists > nn_d[:, k - 1]

        if torch.any(newly_converged):
            # Permute
            r = torch.arange(converged_index)
            new_order = torch.cat([r[~newly_converged], r[newly_converged]])

            ordering[:converged_index] = ordering[new_order]
            neighbors_dist[:converged_index] = neighbors_dist[new_order]
            neighbors_index[:converged_index] = neighbors_index[new_order]
            lines_idx[:converged_index] = lines_idx[new_order]
            lines_d[:converged_index] = lines_d[new_order]
            batch[:converged_index] = batch[new_order]
            sigmas[:converged_index] = sigmas[new_order]

            converged_index = len(r[~newly_converged])
            del r
            del new_order

        del points
        del points_index
        del diff
        del mask
        del unsorted_dists
        del nn_d
        del next_dists
        del newly_converged

    reverse_ordering = torch.argsort(ordering)
    del ordering
    del batch
    del sigmas
    del lines_d
    del lines_idx
    if verbose:
        return neighbors_index[reverse_ordering], neighbors_dist[
            reverse_ordering], i.cpu().numpy(), c.cpu().numpy()
    return neighbors_index[reverse_ordering], neighbors_dist[reverse_ordering]


def get_indices_between_boundaries(t_batch, t_traj, limit_bound_idx):
    limit, boundaries, idx = limit_bound_idx
    if idx is None:
        idx = torch.bucketize(t_batch, t_traj)
    closest_inf_b = torch.clamp(torch.bucketize(idx, boundaries) - 1, min=0)
    idx = torch.max(idx - limit, boundaries[closest_inf_b]).unsqueeze(1)
    return idx + torch.arange(0, limit, device=t_batch.device).unsqueeze(0)


def setup_tensors(batch, k):
    device = batch.device
    neighbors_dist = torch.full((batch.shape[0], k),
                                float("inf"),
                                device=device)
    neighbors_index = torch.zeros((batch.shape[0], k), device=device).long()
    converged_index = batch.shape[0]
    ordering = torch.arange(batch.shape[0], device=device)

    return neighbors_dist, neighbors_index, converged_index, ordering
