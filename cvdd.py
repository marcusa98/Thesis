import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
from clustpy.data import load_iris, load_mnist, load_kmnist, load_fmnist, load_pendigits, load_coil20
from DBCV_Lena import MST_Edges
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import pairwise_distances


def build_graph(Edges): 
    # function to represent graph as dictionary of lists
    graph = defaultdict(list)
    for edge in Edges:
        u, v, w = edge
        graph[u].append((v, w))
        graph[v].append((u, w))
    return graph

def dfs(original_node, current_node, graph, visited, minmax_matrix, current_max):
    # depth first search algorithm to compute max weights for each possible path
    visited[current_node] = True
    for neighbor, weight in graph[current_node]:
        # print("original node", original_node)
        # print("neighbor", neighbor)
        if not visited[neighbor]:
            current_max = max(current_max, weight)
            # if neighbor > original_node:
            minmax_matrix[int(original_node), int(neighbor)] = current_max
            # else:
            #     pass
            dfs(original_node, neighbor, graph, visited, minmax_matrix, current_max)

def MinMaxDists(Edges):
    """
    Edges...np.array of shape n-1 x 3 with columns outgoing node, incoming node, path weight

    returns dictionary with MinMax Dists for all paths
    """
    graph = build_graph(Edges)
    nodes = np.unique(Edges[:, :2])

    # Store all MinMax distances
    minmax_matrix = np.zeros((len(nodes), len(nodes)), dtype=np.float32)
    #minmax_matrix = np.zeros((len(nodes), len(nodes)))
    # Perform DFS from each node to find maximum edge weights to other nodes
    for node in nodes:
        #print(node)
        visited = defaultdict(bool)
        #visited = np.zeros((nodes.max() + 1), dtype=bool) 
        dfs(node, node, graph, visited, minmax_matrix, 0)
        #print(minmax_matrix)
        
    return minmax_matrix


def cvdd(X, y, k = 5, distance = euclidean):
    """
    Compute CVDD of a clustering solution.

    Parameters:
        X (array-like): Dataset.
        y (array-like): Cluster labels for each data point.
        k (integer): number of nearest neighbours to defined a corepoint
    Returns:
        float: CVDD of the clustering solution.
    """  
    n = len(y)
    stats = {}
    n_clusters = len(np.unique(y))

    # get k Nearest neighbors
    knn = NearestNeighbors(n_neighbors =  k + 1).fit(X)
    distances, _ = knn.kneighbors(X)

    # compute density estimation (Def. 2)
    stats["Den"] = np.mean(distances[:, 1:], axis=1)
    
    # compute outlier factor (Def. 3)
    stats["fDen"] = stats["Den"] / np.max(stats["Den"])

    # set drD to euclidean distances in the beginning
    #drD = squareform(pdist(X))
    print("start computing pairwise distances")

    drD = pairwise_distances(X, X)

    print("finished computing pairwise distances")

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

    # fRel = np.zeros((n, n))
    # # initialize Matrix for sum of Den(x_i) and Den(x_j)
    # nD = np.zeros((n, n))
    # for i in trange(n):
    #     #print(stats["Den"][i])
    #     for j in range(n):
    #         if i < j:

    #             # compute nD(x_i, x_j)
    #             nD[i, j] = stats["Den"][i] + stats["Den"][j]
    #             nD[j, i] = stats["Den"][j] + stats["Den"][i]
            
    #             # Relative Density (Def. 5)
    #             Rel_ij = stats["Den"][i] / stats["Den"][j]
    #             Rel_ji = stats["Den"][j] / stats["Den"][i]
            
    #             # mutual Density factor (Def. 6)
    #             fRel[i, j] = 1 - np.exp(-(Rel_ij + Rel_ji - 2))
    #             fRel[j, i] = 1 - np.exp(-(Rel_ji + Rel_ij - 2))

    # relD = fRel * nD


    drD += relD
    
    G = {
        "no_vertices": n,
        "MST_edges": np.zeros((n - 1, 3)),
        "MST_degrees": np.zeros((n), dtype=int),
        "MST_parent": np.zeros((n), dtype=int),
    }


    # compute MST using Lena's implementation of Prim algorithm
    Edges_drD, _ = MST_Edges(G, 0, drD)   

    conD = MinMaxDists(Edges_drD)

    # use broadcasting and matrix matrix operations instead of loops
    fDen_reshaped = stats["fDen"].reshape(-1, 1)

    DD = ((fDen_reshaped * fDen_reshaped.T) ** 0.5) * conD


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
        #inter_dists = squareform(pdist(X[np.where(y == cluster)[0],:]))
        inter_dists = pairwise_distances(X[np.where(y == cluster)[0],:], X[np.where(y == cluster)[0],:])


        # compute MST using Lena's implementation of Prim algorithm
        Edges_pD, _ = MST_Edges(G, 0, inter_dists)

        pD = MinMaxDists(Edges_pD)

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




if __name__ == "__main__":
    from ucimlrepo import fetch_ucirepo 
    from sklearn.preprocessing import LabelEncoder
    from time import time
    from tqdm import trange
    from scipy.sparse import csr_matrix
    from sklearn.metrics.pairwise import pairwise_distances
    import pandas as pd
    label_encoder = LabelEncoder()
    import reference_prim_mst

    # # fetch dataset
    # ionosphere = fetch_ucirepo(id=52)

    # # data (as pandas dataframes)
    # X_ion = ionosphere.data.features
    # y_ion = ionosphere.data.targets

    # y_ion = label_encoder.fit_transform(y_ion['Class'])

    #time wiht old version -> 0.67 sec
    #time with new version -> 0.45 sec
    # start = time()
    # print(cvdd(np.array(X_ion), np.array(y_ion)))
    # stop = time()
    # print(stop-start)

    # iris = load_iris()
    # X_iris, y_iris = iris['data'], iris['target']
    # start = time()
    # print(cvdd(np.array(X_iris), np.array(y_iris)))
    # stop = time()
    # print(stop-start)

    # print(X_iris)

    mnist = load_mnist()

    X_mnist = mnist["data"]
    Y_mnist = mnist["target"]

    combined_data = list(zip(X_mnist, Y_mnist))
    np.random.shuffle(combined_data)

    # Unpack the shuffled data
    X_mnist, Y_mnist = zip(*combined_data)

    # Convert back to numpy arrays
    X_mnist = np.array(X_mnist)[:10000]
    y_mnist = np.array(Y_mnist)[:10000]


    cvdd(X_mnist, y_mnist)

    # mnist_dists = pairwise_distances(X_mnist,X_mnist)
    # mnist_dists_df = pd.DataFrame(mnist_dists)

    # # Write the DataFrame to a CSV file
    # mnist_dists_df.to_csv('mnist_dists.csv', index=False, header= False)

    # print("mnist done")



    kmnist = load_kmnist()

    X_kmnist = kmnist['data']
    y_kmnist = kmnist['target']

    combined_data = list(zip(X_kmnist, y_kmnist))
    np.random.shuffle(combined_data)

    # Unpack the shuffled data
    X_kmnist, y_kmnist = zip(*combined_data)

    # Convert back to numpy arrays
    X_kmnist = np.array(X_kmnist)[:10000]
    y_kmnist = np.array(y_kmnist)[:10000]

    cvdd(X_kmnist, y_kmnist)
    # kmnist_dists = pairwise_distances(X_kmnist,X_kmnist)
    # kmnist_dists_df = pd.DataFrame(kmnist_dists)

    # # Write the DataFrame to a CSV file
    # kmnist_dists_df.to_csv('kmnist_dists.csv', index = False, header = False)

    # print("kmnist done")



    fmnist = load_fmnist()

    X_fmnist = fmnist["data"]
    Y_fmnist = fmnist["target"]

    combined_data = list(zip(X_fmnist, Y_fmnist))
    np.random.shuffle(combined_data)

    # Unpack the shuffled data
    X_fmnist, Y_fmnist = zip(*combined_data)

    # Convert back to numpy arrays
    X_fmnist = np.array(X_fmnist)[:10000]
    y_fmnist = np.array(Y_fmnist)[:10000]

    cvdd(X_fmnist, y_fmnist)

    # fmnist_dists = pairwise_distances(X_fmnist,X_fmnist)
    # fmnist_dists_df = pd.DataFrame(fmnist_dists)

    # # Write the DataFrame to a CSV file
    # fmnist_dists_df.to_csv('fmnist_dists.csv', index = False, header = False)

    # print("fmnist done")

    # mst = reference_prim_mst.prim_mst(dists, 0)

    # n = X.shape[0]
    # G = {
    #         "no_vertices": n,
    #         "MST_edges": np.zeros((n - 1, 3)),
    #         "MST_degrees": np.zeros((n), dtype=int),
    #         "MST_parent": np.zeros((n), dtype=int),
    #     }

    # mst_lena, _ = MST_Edges(G, 0, dists)

    # print(mst.shape)
    # print(mst_lena.shape)


    # start = time()
    #print(cvdd(X, y))
    # stop = time()

    # print(stop-start)
    # no csr_matrix: 
    # mnist[:10000] 
    # total time: 127 sec
    # time until drD Matrix is finished: 25 sec

    # csr_matrix:
    # mnist[:10000]
    # total time: 121.3 sec
    # time until drD Matrix is finished: 17 sec



    # start = time()
    # dists1 = squareform(pdist(X))
    # # print(dists1 / 2 ** 0.5 * dists1)
    # stop = time()
    # print("squareform(pdist(X)) with no csr_matrix", stop-start)


    # #### WINNER for MNIST distance matrix, time: 63 seconds #####
    # #### WINNER for Synth Low distance matrix, time: 0.358 seconds #####
    # #### WINNER for Synth High distance matrix, time: 0.244 seconds #####
    # #### WINNER for Pendigits distance matrix, time: 1.332 seconds #####
    # #### WINNER for Coil-20 distance matrix, time: 0.176 seconds #####

    # start = time()
    # dists3 = pairwise_distances(X, X)
    # # print(dists2 / 2 ** 0.5 * dists2)
    # stop = time()
    # print("pairwise_distances(X) with no csr_matrix", stop - start)

    # print(dists3)

    # start = time()
    # dists4 = pairwise_distances(csr_matrix(X), csr_matrix(X))
    # # print(dists2 / 2 ** 0.5 * dists2)
    # stop = time()
    # print("pairwise_distances(X) with csr_matrix", stop - start)








    # tft = dists1_sub + np.array([1], dtype=np.float32)

    # print(tft)
    # print(tft.dtype)

    # print(dists1_sub.dtype())
    # print(dists2_sub.dtype())

    # Mnist[:10000]: time with old version -> 184 sec until drD and 521 sec total
    # Mnist[:10000]: time with relD vectorized version -> 334 sec total
    # Mnist[:10000]: time with relD and conD vectorized version -> 289 sec total
    # start = time()
    # print(cvdd(X, y))
    # stop = time()
    # print(stop-start)




    import numpy as np
    from collections import defaultdict

    def build_graph1(Edges): 
        # function to represent graph as dictionary of lists
        graph = defaultdict(list)
        for edge in Edges:
            u, v, w = edge
            graph[u].append((v, w))
            graph[v].append((u, w))
        return graph

    def dfs1(original_node, current_node, graph, visited, minmax_matrix, current_max):
        # depth first search algorithm to compute max weights for each possible path
        visited[current_node] = True
        original_max = current_max # reset current_max for every new immediate neighbor
        for neighbor, weight in graph[current_node]:
            if not visited[neighbor]:
                current_max = max(original_max, weight)
                minmax_matrix[int(original_node), int(neighbor)] = current_max
                dfs1(original_node, neighbor, graph, visited, minmax_matrix, current_max)

    def MinMaxDists1(Edges):
        """
        Edges...np.array of shape n-1 x 3 with columns outgoing node, incoming node, path weight

        returns dictionary with MinMax Dists for all paths
        """
        graph = build_graph1(Edges)
        nodes = np.unique(Edges[:, :2])

        print(nodes)
        # Store all MinMax distances
        minmax_matrix = np.zeros((len(nodes), len(nodes)))
        # Perform DFS from each node to find maximum edge weights to other nodes
        for node in nodes:
            #visited = defaultdict(bool)
            visited = np.zeros(nodes.max() + 1, dtype=bool)
            dfs1(node, node, graph, visited, minmax_matrix, 0)
        
        return minmax_matrix

    # Example usage
    # Edges = np.array([
    #     [0, 1, 4],
    #     [0, 2, 3],
    #     [1, 3, 2],
    #     [1, 4, 6],
    #     [2, 5, 1],
    #     [2, 6, 5],
    #     [3, 7, 7],
    #     [4, 8, 8],
    #     [5, 9, 4],
    #     [6, 10, 3],
    #     [7, 11, 2],
    #     [8, 12, 6],
    #     [9, 13, 5],
    #     [10, 14, 1]
    # ])

    # start = time()
    # min_max_dists = MinMaxDists1(Edges)
    # stop = time()
    # print(min_max_dists)
    # print("Matrix weights", stop-start)

    # start = time()
    # min_max_dists_original = MinMaxDists(Edges)
    # stop = time()
    # print(min_max_dists_original)
    # print("dictionary weights", stop-start)


    #print(max(Edges[0, 2], 1))


    
    # def MST_Edges(G, start, G_edges_weights):
    #     """
    #     Calculate minimal spanning tree.
    #     Returning edges of MST and degrees for each node.

    #     :param G: dict including information
    #     :param start: starting point for Prim
    #     :param G_edges_weights: matrix including edge weights
    #     :return: Edg, Degr
    #     """
    #     # Prims algorithm
    #     # distance array
    #     d = []
    #     # array storing if already visited
    #     intree = []
    #     # initialize
    #     for i in range(G["no_vertices"]):
    #         # add first no node is already visited
    #         intree.append(0)
    #         # the distance to all vertices is inf
    #         d.append(np.inf)
    #         # Parents are set to themselves
    #         G["MST_parent"][i] = int(i)
    #     # starting at start
    #     d[start] = 0
    #     # pointer
    #     v = start
    #     counter = -1
    #     # count until we connected all nodes
    #     while counter < G["no_vertices"] - 2:
    #         # we add v to the 'visited'
    #         intree[v] = 1
    #         dist = np.inf
    #         # for every node
    #         for w in range(G["no_vertices"]):
    #             # if the node is not already in the visited elements and is not the same as the one we want to check
    #             if (w != v) & (intree[w] == 0):
    #                 # we look at the distance
    #                 weight = G_edges_weights[v, w]
    #                 # if the distance is smaller than the distance that connects w currently we update
    #                 if d[w] > weight:
    #                     d[w] = weight
    #                     G["MST_parent"][w] = int(v)
    #                 # if the distance is smaller than current dist we update dist
    #                 if dist > d[w]:
    #                     dist = d[w]
    #                     next_v = w
    #         # update counter for dict
    #         counter = counter + 1
    #         # the outgoing node is the parent
    #         outgoing = G["MST_parent"][next_v]
    #         # incoming is the current vertex
    #         incoming = next_v
    #         # add a new edge
    #         G["MST_edges"][counter, :] = [
    #             outgoing,
    #             incoming,
    #             G_edges_weights[outgoing, incoming],
    #         ]
    #         # update degrees for outgoing and incoming
    #         G["MST_degrees"][G["MST_parent"][next_v]] = (
    #                 G["MST_degrees"][G["MST_parent"][next_v]] + 1
    #         )
    #         G["MST_degrees"][next_v] = G["MST_degrees"][next_v] + 1
    #         # set next vertex
    #         v = next_v
    #     # extract Edges and degrees from the dict
    #     Edg = G["MST_edges"]
    #     Degr = G["MST_degrees"]
    #     return Edg, Degr


    # def MST_Edges(G, start, G_edges_weights):
    #     """
    #     Calculate minimal spanning tree.
    #     Returning edges of MST and degrees for each node.

    #     :param G: dict including information
    #     :param start: starting point for Prim
    #     :param G_edges_weights: matrix including edge weights
    #     :return: Edg, Degr
    #     """
    #     no_vertices = G["no_vertices"]
    #     MST_parent = G["MST_parent"]
    #     MST_edges = G["MST_edges"]
    #     MST_degrees = G["MST_degrees"]

    #     # Prims algorithm
    #     # distance array
    #     d = np.full(no_vertices, np.inf)
    #     # array storing if already visited
    #     intree = np.zeros(no_vertices, dtype=bool)
    #     # initialize parents
    #     MST_parent[:] = np.arange(no_vertices)
        
    #     # starting at start
    #     d[start] = 0
    #     # pointer
    #     v = start
    #     counter = -1
    #     # count until we connected all nodes
    #     while counter < no_vertices - 2:
    #         # we add v to the 'visited'
    #         intree[v] = True
    #         dist = np.inf
    #         next_v = -1
            
    #         # for every node
    #         for w in range(no_vertices):
    #             # if the node is not already in the visited elements and is not the same as the one we want to check
    #             if not intree[w] and w != v:
    #                 # we look at the distance
    #                 weight = G_edges_weights[v, w]
    #                 # if the distance is smaller than the distance that connects w currently we update
    #                 if d[w] > weight:
    #                     d[w] = weight
    #                     MST_parent[w] = v
    #                 # if the distance is smaller than current dist we update dist
    #                 if dist > d[w]:
    #                     dist = d[w]
    #                     next_v = w
            
    #         if next_v == -1:
    #             # No next vertex found, which means the graph is not fully connected
    #             break

    #         # update counter for dict
    #         counter += 1
    #         # the outgoing node is the parent
    #         outgoing = MST_parent[next_v]
    #         # incoming is the current vertex
    #         incoming = next_v
    #         # add a new edge
    #         MST_edges[counter, :] = [outgoing, incoming, G_edges_weights[outgoing, incoming]]
    #         # update degrees for outgoing and incoming
    #         MST_degrees[outgoing] += 1
    #         MST_degrees[incoming] += 1
    #         # set next vertex
    #         v = next_v
        
    #     # extract Edges and degrees from the dict
    #     Edg = MST_edges[:counter + 1]
    #     Degr = MST_degrees
    #     return Edg, Degr


    n = 30000
    dists = np.random.random((n, n))

    G = {
        "no_vertices": n,
        "MST_edges": np.zeros((n - 1, 3)),
        "MST_degrees": np.zeros((n), dtype=int),
        "MST_parent": np.zeros((n), dtype=int),
    }


    start = time()
    Edges_drD, _ = MST_Edges(G, 0, dists)  
    stop = time()

    print(stop - start)

    #time n = 5000, 3.77 sec
    #time n = 10000, 14.45 sec
    #time n = 30000, 142 sec