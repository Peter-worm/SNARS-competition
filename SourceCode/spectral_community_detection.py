import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from numba import njit

def custom_kmeans(X, n_clusters, iterations=1000, eps=1e-4):
    """
    Custom implementation of the KMeans algorithm.

    Args:
        X : numpy.ndarray
            The data to cluster, shape (n_samples, n_features).
        n_clusters : int
            The number of clusters.
        max_iter : int
            Maximum number of iterations for the algorithm.
        tol : float
            Tolerance for convergence.

    Returns:
        tuple
            A tuple (labels, centroids):
                labels: numpy.ndarray, shape (n_samples,)
                    Cluster assignments for each sample.
                centroids: numpy.ndarray, shape (n_clusters, n_features)
                    Cluster centroids.
    """
    n_samples, _ = X.shape
    indices = np.random.choice(n_samples, n_clusters, replace=False)
    centroids = X[indices]

    for iteration in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == k].mean(axis=0) if np.any(labels == k) else centroids[k] for k in range(n_clusters)])
        if np.linalg.norm(new_centroids - centroids) < eps:
            break
        centroids = new_centroids

    return labels, centroids


def spectral_community_detection(adj_matrix, number_of_communities, use_custom_kmeans = False):
    """
    Implements the spectral method for community detection.

    Args:
        adj_matrix : np.array
            np.ndarray representing the adjacency matrix
        number_of_communities : int
            The number of communities to detect.
        use_custom_kmeans: bool
            parameter telling witch version of algorithm should be used

    Returns:
        dict
            A dictionary with nodes as keys and community as values.
    """
    G = nx.from_numpy_array(adj_matrix)
    laplacian_matrix = nx.normalized_laplacian_matrix(G).toarray()

    _, eigvecs = np.linalg.eigh(laplacian_matrix)
    eigvecs_subset = eigvecs[:, 1:number_of_communities]

    if use_custom_kmeans:
        labels = custom_kmeans(eigvecs_subset, n_clusters=number_of_communities)[0]
    else:
        kmeans = KMeans(number_of_communities)
        kmeans.fit(eigvecs_subset)
        labels = kmeans.labels_

    communities = {node: label for node, label in zip(G.nodes(), labels)}

    return list(communities.values())