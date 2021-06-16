import torch
from trajectory_nearest_neighbors import topology


def test_proper_padding():
    input = torch.tensor([1, 2, 1])
    out = topology.proper_padding(input, 7)
    assert torch.all(out == torch.tensor([1, 1, 1, 2, 1, 1, 1]))
    out = topology.proper_padding(input, 6)
    assert torch.all(out == torch.tensor([1, 1, 1, 2, 1, 1]))


def test_trajectory_creator():
    X = torch.randn(100, 4)
    y = torch.randn(100, 2)
    tnti = {"a": torch.arange(50), "b": torch.arange(50, 100)}
    tc = topology.TrajectoryCreator(X, y, tnti, 5)
    print(tc)
