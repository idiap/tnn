import torch
from trajectory_nearest_neighbors import tnn, topology


def test_limit_simple():
    limit_bound_idx = (5, torch.tensor([0, 10]), None)
    batch = torch.tensor([2, 7])
    t = torch.arange(0, 21)

    results = torch.tensor([[0, 1, 2, 3, 4], [2, 3, 4, 5, 6]])

    assert torch.all(results == tnn.get_indices_between_boundaries(
        batch, t, limit_bound_idx))


def test_setup_tensor():
    batch = torch.randn((10, 4))
    k = 7
    neighbors_dist, neighbors_index, converged_index, ordering = tnn.setup_tensors(
        batch, k)

    assert neighbors_dist.shape == (10, k)
    assert neighbors_index.shape == (10, k)
    assert converged_index == 10
    assert ordering[-1] == 9


def test_distance_to_segments():
    X = torch.randn(100, 4)
    y = torch.randn(100, 2)
    tnti = {"a": torch.arange(50), "b": torch.arange(50, 100)}
    tc = topology.TrajectoryCreator(X, y, tnti, 5)
    n_traj = tc.A.shape[0]
    time_window = 0.5
    batch = torch.randn(50, 4)
    s = torch.randn(50, 4)
    dd, ii = tnn.distance_to_segments(tc, batch, time_window, s)

    assert dd.shape == (50, n_traj)
    assert ii.shape == (50, n_traj)


def test_distance_to_segments_with_lim():
    X = torch.randn(100, 4)
    y = torch.randn(100, 2)
    tnti = {"a": torch.arange(50), "b": torch.arange(50, 100)}
    tc = topology.TrajectoryCreator(X, y, tnti, 5)
    n_traj = tc.A.shape[0]
    time_window = 0.5
    batch = torch.randn(50, 4)
    s = torch.randn(50, 4)
    limit_bound_idx = (5, torch.tensor([0, n_traj]), None)
    dd, ii = tnn.distance_to_segments_with_lim(tc, batch, time_window, s,
                                               limit_bound_idx)

    assert dd.shape == (50, 5)
    assert ii.shape == (50, 5)


def test_query():
    X = torch.randn(100, 4)
    y = torch.randn(100, 2)
    tnti = {"a": torch.arange(50), "b": torch.arange(50, 100)}
    tc = topology.TrajectoryCreator(X, y, tnti, 5)
    n_traj = tc.A.shape[0]
    time_window = 0.5
    batch = torch.randn(50, 4)
    s = torch.randn(50, 4)

    dd, ii = tnn.query(tc, X, batch, time_window, s=s, k=7)

    assert dd.shape == (50, 7)
    assert ii.shape == (50, 7)

    limit_bound_idx = (5, torch.tensor([0, n_traj]), None)
    dd, ii = tnn.query(tc,
                       X,
                       batch,
                       time_window,
                       s=s,
                       k=7,
                       limit_bound_idx=limit_bound_idx)

    assert dd.shape == (50, 7)
    assert ii.shape == (50, 7)