import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
from clustpy.data import load_iris, load_mnist, load_kmnist, load_fmnist, load_pendigits, load_coil20
from DBCV_Lena import MST_Edges
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import pairwise_distances
import sys
from tqdm import tqdm
from distance_metric import get_dc_dist_matrix


def build_graph(Edges): 
    # function to represent graph as dictionary of lists
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
    Edges...np.array of shape n-1 x 3 with columns outgoing node, incoming node, path weight

    returns dictionary with MinMax Dists for all paths
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


def cvdd(X, y, k = 5, dc_distance = False):
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
    if dc_distance:
        drD = get_dc_dist_matrix(X, k)
    else:
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
        if dc_distance:
            inter_dists = get_dc_dist_matrix(X[np.where(y == cluster)[0],:])
        else:
            inter_dists = pairwise_distances(X[np.where(y == cluster)[0],:], X[np.where(y == cluster)[0],:])

        # compute MST using Lena's implementation of Prim algorithm
        Edges_pD, _ = MST_Edges(G, 0, inter_dists)

        pD = MinMaxDists(Edges_pD, inter_dists)

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