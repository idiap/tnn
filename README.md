# Trajectory Nearest Neighbors

Author : [Arnaud Pannatier \<arnaud.pannatier@idiap.ch\>](mailto:arnaud.pannatier@idiap.ch)
For any questions/remarks about this work or about my research, feel free to contact me.


## Abstract
  This is a novel algorithm for fetching nearest neighbors in a data set whose elements are organized along smooth trajectories that can be approximated with piece-wise linear structures.
  We introduce an efficient and exact strategy that can be implemented with algebraic tensorial operations and is consequently very adapted to modern GPU-based computing infrastructure.

  This method can be used with a scalable Euclidean metric and allows to mask some data points along one dimension.
  When applied, this method is more efficient than plain Euclidean $k$-NN and other well-known data selection methods like KDTrees, and provides a several-fold speed-up.

  We demonstrate the efficiency of our approach in a machine learning pipeline to forecast high-altitude wind speed from live data measured by planes. We provide an implementation in PyTorch and a novel data set to allow the replication of empirical results.

  A publication is pending review at [ICONIP 2021](https://iconip2021.apnns.org/).


## How to use ? 

This algorithm work best if your dataset is organized in trajectories.
By default it assumes 4D point, and do the masking along the 4th dimension (time).
If you need to adapt it you can contact me directly, and I will provide guidance.
A full description of the algorithm can be found in the publication.

### Split the dataset into trajectories
First you need to convert your dataset into objects that can be processed by the algorithm.
The code for that part can be found in `topology.py`

You can find a basic example here:
``` python
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

``` python
batch = torch.randn(50,4)
time_window = 0.1
scaling = torch.randn(50,4) # Use torch.ones if you don't want scaling

dd, ii = tnn.query(tc, X, batch, time_window, s=s, k=7)
```
### Limit the number of segments 

If you have a huge number of lines, you might want to fetch the points in only 
a subpart of them. The algorithm will be much faster but might omit points that
were contained in lines that were further away.
If your dataset is constituted of chunk of points and that you want to limit the query to a given chunk you need to provide the boundaries of those chunks and the corresponding indexes. If you don't need that just pass (0, n_lines) as boundaries.

To do both you need to give to the algorithm a tupple `(limit, boundaries, index)`
Here is an example showing the limitation to the 2 previous segments

``` python
batch = torch.randn(50,4)
time_window = 0.1
scaling = torch.randn(50,4) # Use torch.ones if you don't want scaling

limit_bound_idx = (5, torch.tensor([0, n_traj]), None)
dd, ii = tnn.distance_to_segments_with_lim(tc, batch, time_window, s,
                                            limit_bound_idx)
```
