# CUDA-k-NN

The k-nearest neighbor algorithm (k-NN) is a widely used machine learning algorithm used for both classification and regression. k-NN algorithms are used in many research and industrial domains such as 3-dimensional object rendering, content-based image retrieval, statistics (estimation of entropies and divergences), biology (gene classification), etc. The processing time of the kNN search still remains the bottleneck in many application domains, especially in high dimensional spaces. In this work, we try to address this processing time issue by performing the kNN search using the GPU.


Inputs values of the k-NN algorithm:

- Set of reference points.
- Set of query points.
- Parameter k corresponding to the number of neighbors to search for.

For each query point, the k-NN algorithm locates the k closest points (k nearest neighbors) among the reference points set. 
The algorithm returns: 
- the indexes (positions) of the k nearest points in the reference points set
- the k associated Euclidean distances
