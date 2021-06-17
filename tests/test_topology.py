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

    assert tc.X.shape == X.shape
    assert tc.y.shape == y.shape
    assert len(tc.trajectories) == 2
    assert len(tc.lines) > 2
    n_traj = len(tc.lines)
    assert tc.A.shape == (n_traj, 3)
    assert tc.B.shape == (n_traj, 3)
    assert tc.TA.shape == (n_traj, )
    assert tc.TB.shape == (n_traj, )
    assert tc.U.shape == (n_traj, 3)
    assert tc.E.shape == (n_traj, )
    assert tc.line_idx_matrix.nelement() >= 100

    assert torch.all(tc.TA[:-1] <= tc.TA[1:])
    assert torch.all(tc.TA <= tc.TB)