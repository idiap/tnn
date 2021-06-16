import torch
from trajectory_nearest_neighbors import tnn


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
    pass