import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
#from clustpy.data import load_iris, load_mnist, load_kmnist, load_fmnist, load_pendigits, load_coil20
from cvi.mst import MST_Edges
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm

def build_graph(Edges): 
    """
    Construct a graph from an edge list representation.

    This function takes a list of edges and builds a graph represented as a dictionary of adjacency lists.
    Each key in the dictionary corresponds to a node in the graph, and the associated value is a list of
    tuples where each tuple represents a neighboring node and the weight of the edge connecting the nodes.

    Parameters:
    Edges (np.ndarray or list of tuples): A 2D array or list where each entry represents an edge with three components:
                                           - Outgoing node (int)
                                           - Incoming node (int)
                                           - Edge weight (float)

    Returns:
    dict: A dictionary where keys are nodes and values are lists of tuples. Each tuple contains:
          - Neighbor node (int)
          - Edge weight (float)

    Example:
    >>> edges = np.array([[0, 1, 10], [1, 2, 5], [2, 0, 15]])
    >>> build_graph(edges)
    {0: [(1, 10), (2, 15)], 1: [(0, 10), (2, 5)], 2: [(1, 5), (0, 15)]}
    """
    graph = defaultdict(list)
    for edge in Edges:
        u, v, w = edge
        graph[u].append((v, w))
        graph[v].append((u, w))
    return graph

# def dfs(original_node, current_node, graph, visited, minmax_matrix, current_max):
#     # depth first search algorithm to compute max weights for each possible path
#     visited[current_node] = True
#     for neighbor, weight in graph[current_node]:
#         # print("original node", original_node)
#         # print("neighbor", neighbor)
#         if not visited[neighbor]:
#             current_max = max(current_max, weight)
#             # if neighbor > original_node:
#             minmax_matrix[int(original_node), int(neighbor)] = current_max
#             # else:
#             #     pass
#             dfs(original_node, neighbor, graph, visited, minmax_matrix, current_max)

def dfs(original_node, graph, visited, minmax_matrix):
    """
    Perform Depth-First Search (DFS) from a given starting node to compute the maximum edge weights.

    This function traverses the graph using DFS starting from `original_node`. For each node visited,
    it updates the `minmax_matrix` with the maximum edge weight encountered along the path from
    `original_node` to that node.

    Parameters:
    original_node (int): The starting node for the DFS traversal.
    graph (dict): A dictionary where keys are nodes and values are lists of tuples. Each tuple
                  represents a neighbor node and the weight of the edge connecting the nodes.
    visited (dict): A dictionary where keys are nodes and values are boolean flags indicating
                    whether a node has been visited.
    minmax_matrix (np.ndarray): A 2D numpy array where the entry at (i, j) is updated with the
                                maximum edge weight encountered on the path from `i` to `j`.

    Returns:
    None: The function updates `minmax_matrix` in place.
    """
    stack = [(original_node, original_node, 0)]  # Stack contains tuples of (original_node, current_node, current_max)
    
    while stack:
        original_node, current_node, current_max = stack.pop()
        
        if not visited[current_node]:
            visited[current_node] = True
            for neighbor, weight in graph[current_node]:
                if not visited[neighbor]:
                    new_max = max(current_max, weight)
                    minmax_matrix[int(original_node), int(neighbor)] = new_max
                    stack.append((original_node, neighbor, new_max))



def MinMaxDists(Edges, minmax_matrix):
    """
    Compute the maximum edge weights for all pairs of nodes on a MST.

    This function builds a graph from the provided edge list and then uses Depth-First Search (DFS)
    to determine the maximum edge weight encountered on the path between each pair of nodes. The result
    is stored in `minmax_matrix`, where the entry at (i, j) represents the maximum edge weight on any
    path from node `i` to node `j`.

    Parameters:
    Edges (np.ndarray): A numpy array of shape (n-1, 3) where each row represents an edge with columns:
                        - Outgoing node (int)
                        - Incoming node (int)
                        - Path weight (float)
    minmax_matrix (np.ndarray): A 2D numpy array of shape (num_nodes, num_nodes) that will be filled
                                with the maximum edge weights for all pairs of nodes.

    Returns:
    np.ndarray: The `minmax_matrix` with updated maximum edge weights for all paths between nodes.
    """
    graph = build_graph(Edges)
    nodes = np.unique(Edges[:, :2]).astype(int)

    # Store all MinMax distances
    minmax_matrix.fill(0)
    # Perform DFS from each node to find maximum edge weights to other nodes
    for node in tqdm(nodes):
        visited = defaultdict(bool)
        #dfs(node, node, graph, visited, minmax_matrix, 0)
        dfs(node, graph, visited, minmax_matrix)

        
    return minmax_matrix


def cvdd(X, y, k = 5):
    """
    Compute CVDD of a clustering solution.

    Parameters:
        X (array-like): Dataset.
        y (array-like): Cluster labels for each data point.
        k (integer): number of nearest neighbours to define a corepoint
    Returns:
        float: CVDD of the clustering solution.
    """  
    n = len(y)
    stats = {}
    
    # noise is not a cluster
    if any(y == -1):
        n_clusters = len(np.unique(y)) - 1
    else:
        n_clusters = len(np.unique(y))

    # get k Nearest neighbors
    knn = NearestNeighbors(n_neighbors =  k + 1).fit(X)
    distances, _ = knn.kneighbors(X)

    # compute density estimation (Def. 2)
    stats["Den"] = np.mean(distances[:, 1:], axis=1)
    
    # compute outlier factor (Def. 3)
    stats["fDen"] = stats["Den"] / np.max(stats["Den"])

    # set drD to euclidean distances in the beginning
    drD = pairwise_distances(X, X)
        
    # use broadcasting to speed up computation
    den_reshaped = stats["Den"].reshape(-1,1)

    nD = den_reshaped + den_reshaped.T
    np.fill_diagonal(nD, 0.0)

    # Relative Density (Def. 5) 
    Rel_ij = den_reshaped / den_reshaped.T
    np.fill_diagonal(Rel_ij, 0.0)
    Rel_ji = np.transpose(Rel_ij)

    # mutual Density factor (Def. 6)
    fRel = 1 - np.exp(-(Rel_ij + Rel_ji - 2))
    relD = fRel * nD

    # free up some memory
    del Rel_ij, Rel_ji, fRel, nD

    drD += relD
    
    # clean up memory
    del relD

    G = {
        "no_vertices": n,
        "MST_edges": np.zeros((n - 1, 3)),
        "MST_degrees": np.zeros((n), dtype=int),
        "MST_parent": np.zeros((n), dtype=int),
    }


    # compute MST using Lena's implementation of Prim algorithm
    Edges_drD, _ = MST_Edges(G, 0, drD)   

    conD = MinMaxDists(Edges_drD, drD)

    # use broadcasting and matrix matrix operations instead of loops
    fDen_reshaped = stats["fDen"].reshape(-1, 1)

    DD = ((fDen_reshaped * fDen_reshaped.T) ** 0.5) * conD

    # clean up memory
    del conD

    # Compute separations between one cluster and all others (Def. 11)
    seps = []

    for i in range(n_clusters):
            seps.append(np.min(DD[np.ix_(np.where(y == i)[0], np.where(y != i)[0])]))

    coms = []

    for cluster in range(n_clusters):

        cluster_size = sum(y == cluster)

        G = {
            "no_vertices": cluster_size,
            "MST_edges": np.zeros((cluster_size - 1, 3)),
            "MST_degrees": np.zeros((cluster_size), dtype=int),
            "MST_parent": np.zeros((cluster_size), dtype=int),
        }

        # compute all pairwise distances inside a cluster
        intra_dists = pairwise_distances(X[np.where(y == cluster)[0],:], X[np.where(y == cluster)[0],:])

        # compute MST using Lena's implementation of Prim algorithm
        Edges_pD, _ = MST_Edges(G, 0, intra_dists)

        pD = MinMaxDists(Edges_pD, intra_dists)

        Mean_Ci = np.mean(pD)
        Std_Ci = np.std(pD)

        # compute compactness (Def. 12)
        #com_Ci = (1/n_paths) * Std_Ci * Mean_Ci
        com_Ci = (1 / cluster_size) * Std_Ci * Mean_Ci

        #print(com_Ci)
        coms.append(com_Ci)



    # compute CVDD (Def. 13)
    res = sum(seps) / sum(coms)

    return res