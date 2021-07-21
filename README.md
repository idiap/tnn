# Trajectory Nearest Neighbors

This code was developed as a part of the **Innosuisse MALAT: Machine Learning for Air Traffic** project, which is a partnership between SkySoft ATM and the Idiap Research Institute.

Main research partner : **Pr. François Fleuret (UNIGE)**

Project manager : **Didier Berling (SkySoft ATM)**

Author : **[Arnaud Pannatier \<arnaud.pannatier@idiap.ch\>](mailto:arnaud.pannatier@idiap.ch) (Idiap Research Institute)**.

For any questions/remarks about this work or my research, feel free to contact the author.

## Abstract

This is a novel algorithm for fetching nearest neighbors in a data set whose elements are organized along smooth trajectories that can be approximated with piece-wise linear structures.
We introduce an efficient and exact strategy that can be implemented with algebraic tensorial operations and is consequently very adapted to modern GPU-based computing infrastructure.

This method can be used with a scalable Euclidean metric and allows to mask data points along one dimension.
When applied, this method is more efficient than plain Euclidean $k$-NN and other well-known data selection methods like KDTrees and provides a several-fold speed-up.

A publication is pending review at [ICONIP 2021](https://iconip2021.apnns.org/).

## Dataset

A dataset containing flight trajectories is available here : [https://www.idiap.ch/en/dataset/skysoft/index_html](https://www.idiap.ch/en/dataset/skysoft/index_html)

## How to use?

This algorithm works best if your dataset is organized in trajectories.
By default, it assumes a dataset made of 4D points and does the masking along the 4th dimension (time).
If you need to adapt it, you can contact me directly, and I will provide guidance.
You can find a complete description of the algorithm in the publication.

### Split the dataset into trajectories

First, you need to convert your dataset into objects that the algorithm can process.
You can find the code for that part in `topology.py`

You can find a basic example here:

```python
# Let X,y be your dataset
X = torch.randn(100, 4) # Position and time
y = torch.randn(100, 2) # vx, vy
# You need to provide a description of the trajectories
# in a dict, the keys corresponds to the name of the objects
# the values corresponds to the index of the points in the trajectories
tnti = {"a": torch.arange(50), "b": torch.arange(50, 100)}
# gives the number of points that each lines will contains
# If the last line don't have enough points it will pad it
points_per_line = 5
# You can then create you trajectories
tc = topology.TrajectoryCreator(X, y, tnti, 5)

# you can find info about the lines
print(tc.lines)
```

### Queries

You can then use that object to query the nearest neighbors of a batch of points, masked before a time window and with a given scaling per dimension

```python
batch = torch.randn(50,4)
time_window = 0.1
scaling = torch.randn(50,4) # Use torch.ones if you don't want scaling

dd, ii = tnn.query(tc, X, batch, time_window, s=s, k=7)
```

### Limit the number of segments

If you have a considerable number of lines, you might want to fetch the points in only
a subpart of them. The algorithm will be much faster but might omit points contained in lines that were further away.
Suppose your dataset comprises chunks of points, and you want to limit the query to a given chunk.
In that case, you need to provide the boundaries of those chunks and the corresponding indexes.
If you don't need that, just pass (0, n_lines) as boundaries.

You need to give the algorithm a tuple `(limit, boundaries, index) to do both.`
Here is an example showing the limitation to the two previous segments

```python
batch = torch.randn(50,4)
time_window = 0.1
scaling = torch.randn(50,4) # Use torch.ones if you don't want scaling

limit_bound_idx = (5, torch.tensor([0, n_traj]), None)
dd, ii = tnn.distance_to_segments_with_lim(tc, batch, time_window, s,
                                            limit_bound_idx)
```
