import numpy as np
from numba import njit


@njit
def find_partition(adj_matrix, gamma=2.0, max_iter=100):
    """
    Greedy algorithm which aims to find a partition which maximizes the modularity
     in Reichardt Bornholdt Potts Model (RB) https://en.wikipedia.org/wiki/Leiden_algorithm ( with gamma )
    The gamma was chosen to be 2 because it seems to work best for provided adj matrices
    In each iteration this algorithm tries to move each node to a community which maximizes would increse the modularity by the largest amount
    The modularity gain and loss are from
    https://stats.stackexchange.com/questions/615770/calculating-modularity-gain-of-switching-a-node-from-one-community-to-another-l
    :param  adj_matrix: dense np.ndarray representing the adjacency matrix
    :param gamma: gamma parameter in the RB model by default set to 2.0
    :param max_iter: the maximum number of iterations of the method by default 100
    :return: np array of ( non consecutive ) community assignments for each node
    """
    n = adj_matrix.shape[0]
    communities = np.arange(n)
    degrees = adj_matrix.sum(axis=1)
    edge_num = adj_matrix.sum()
    resolution_scale = edge_num * 2 / gamma
    community_links = np.zeros(n, dtype=np.int64)
    community_degrees = np.zeros(n, dtype=np.int64)

    for node in range(n):
        community_degrees[communities[node]] += degrees[node]

    for iteration in range(max_iter):
        moved = False

        for node in range(n):
            current_community = communities[node]
            community_links.fill(0)

            for neighbor in range(n):
                weight = adj_matrix[node, neighbor]
                if weight > 0:
                    neighbor_community = communities[neighbor]
                    community_links[neighbor_community] += weight

            best_modularity_gain = 0
            best_community = current_community

            deg_sum_curr = 0
            for neighbor in range(n):
                if communities[neighbor] == current_community:
                    deg_sum_curr += degrees[neighbor]

            modularity_loss = (
                community_links[current_community]
                - (degrees[node] * deg_sum_curr) / resolution_scale
                + degrees[node] * degrees[node] / resolution_scale
            )

            for community in range(n):
                if community_links[community] > 0:
                    deg_sum_checked = 0
                    for neighbor in range(n):
                        if communities[neighbor] == community:
                            deg_sum_checked += degrees[neighbor]

                    modularity_gain = (
                        community_links[community]
                        - (degrees[node] * deg_sum_checked) / resolution_scale
                        - modularity_loss
                    )
                    if modularity_gain > best_modularity_gain:
                        best_modularity_gain = modularity_gain
                        best_community = community

            if best_community != current_community:
                communities[node] = best_community
                community_degrees[best_community] += degrees[node]
                community_degrees[current_community] -= degrees[node]
                moved = True

        if not moved:
            break

    return communities
