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