import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
from clustpy.data import load_iris, load_mnist, load_fmnist
from DBCV_Lena import MST_Edges


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
    original_max = current_max # reset current_max for every new immediate neighbor
    for neighbor, weight in graph[current_node]:
        if not visited[neighbor]:
            current_max = max(original_max, weight)
            minmax_matrix[int(original_node), int(neighbor)] = current_max
            dfs(original_node, neighbor, graph, visited, minmax_matrix, current_max)

def MinMaxDists(Edges):
    """
    Edges...np.array of shape n-1 x 3 with columns outgoing node, incoming node, path weight

    returns dictionary with MinMax Dists for all paths
    """
    graph = build_graph1(Edges)
    nodes = np.unique(Edges[:, :2])

    # Store all MinMax distances
    minmax_matrix = np.zeros((len(nodes), len(nodes)))
    # Perform DFS from each node to find maximum edge weights to other nodes
    for node in nodes:
        visited = defaultdict(bool)
        dfs(node, node, graph, visited, minmax_matrix, 0)
    
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
    drD = squareform(pdist(X))

    # use broadcasting to speed up computation
    array1 = stats["Den"].reshape(-1,1)
    array2 = stats["Den"].reshape(1,-1)

    nD = array1 + array2
    np.fill_diagonal(nD, 0.0)

    # Relative Density (Def. 5) 
    Rel_ij = array1 / array2
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
    array1 = stats["fDen"].reshape(-1, 1)
    array2 = stats["fDen"].reshape(1, -1)

    DD = ((array1 * array2) ** 0.5) * conD


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
        inter_dists = squareform(pdist(X[np.where(y == cluster)[0],:]))

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
    

    label_encoder = LabelEncoder()
    # fetch dataset
    ionosphere = fetch_ucirepo(id=52)

    # data (as pandas dataframes)
    X_ion = ionosphere.data.features
    y_ion = ionosphere.data.targets

    y_ion = label_encoder.fit_transform(y_ion['Class'])

    #time wiht old version -> 0.67 sec
    #time with new version -> 0.45 sec
    start = time()
    print(cvdd(np.array(X_ion), np.array(y_ion)))
    stop = time()
    print(stop-start)

    X_iris, y_iris = load_iris()
    start = time()
    print(cvdd(np.array(X_iris), np.array(y_iris)))
    stop = time()
    print(stop-start)


    X, y = load_mnist()

    X = np.array(X)[:10000]
    y = np.array(y)[:10000]

    # Mnist[:10000]: time with old version -> 184 sec until drD and 521 sec total
    # Mnist[:10000]: time with relD vectorized version -> 334 sec total
    # Mnist[:10000]: time with relD and conD vectorized version -> 289 sec total
    start = time()
    print(cvdd(X, y))
    stop = time()
    print(stop-start)


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

        # Store all MinMax distances
        minmax_matrix = np.zeros((len(nodes), len(nodes)))
        # Perform DFS from each node to find maximum edge weights to other nodes
        for node in nodes:
            visited = defaultdict(bool)
            dfs1(node, node, graph, visited, minmax_matrix, 0)
        
        return minmax_matrix

    # # Example usage
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


    k = 5

    n = len(y_ion)
    stats = {}
    n_clusters = len(np.unique(y_ion))

    # get k Nearest neighbors
    knn = NearestNeighbors(n_neighbors =  k + 1).fit(X_ion)
    distances, _ = knn.kneighbors(X_ion)

    # compute density estimation (Def. 2)
    stats["Den"] = np.mean(distances[:, 1:], axis=1)
    
    # compute outlier factor (Def. 3)
    stats["fDen"] = stats["Den"] / np.max(stats["Den"])

    # set drD to euclidean distances in the beginning
    drD = squareform(pdist(X_ion))

    # use broadcasting to speed up computation
    array1 = stats["Den"].reshape(-1,1)
    array2 = stats["Den"].reshape(1,-1)

    nD = array1 + array2
    np.fill_diagonal(nD, 0.0)

    # Relative Density (Def. 5) 
    Rel_ij = array1 / array2
    np.fill_diagonal(Rel_ij, 0.0)
    Rel_ji = np.transpose(Rel_ij)

    # mutual Density factor (Def. 6)
    fRel = 1 - np.exp(-(Rel_ij + Rel_ji - 2))
    relD = fRel * nD

    drD += relD

    G = {
        "no_vertices": n,
        "MST_edges": np.zeros((n - 1, 3)),
        "MST_degrees": np.zeros((n), dtype=int),
        "MST_parent": np.zeros((n), dtype=int),
    }


    # compute MST using Lena's implementation of Prim algorithm
    Edges_drD, _ = MST_Edges(G, 0, drD)   

    start = time()
    conD = MinMaxDists(Edges_drD)

    DD = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i < j:
                #pass
                #print(conD[str(float(i))][float(j)] == conD[str(float(j))][float(i)])
                DD[i,j] = ((stats["fDen"][i] * stats["fDen"][j]) ** 0.5) * conD[str(float(i))][float(j)]
                DD[j,i] = ((stats["fDen"][i] * stats["fDen"][j]) ** 0.5) * conD[str(float(j))][float(i)]

    stop = time()

    print("non matrix version: ", stop - start, "sec")


    start = time()
    conD = MinMaxDists1(Edges_drD)

    array1 = stats["fDen"].reshape(-1, 1)
    array2 = stats["fDen"].reshape(1, -1)

    DD1 = ((array1 * array2) ** 0.5) * conD
    stop = time()

    print("matrix version: ", stop - start, "sec")

    # print(np.allclose(DD, DD1))

    # X = np.array([[1,2,3,4],[5,6,7,8]])
    # y = np.array([0,3])

    #print(len(np.unique(y)))
    #dcsi(X,y)

    X_iris, y_iris = load_iris()
    
    print("true labels")
    cvdd(X_iris, y_iris)
    print("random labels")
    y_test1 = np.random.randint(3, size = len(y_iris))
    y_test2 = np.random.randint(3, size = len(y_iris))
    y_test3 = np.random.randint(3, size = len(y_iris))
    y_test4 = np.random.randint(3, size = len(y_iris))
    y_test5 = np.random.randint(3, size = len(y_iris))

    cvdd(X_iris, y_test1)
    cvdd(X_iris, y_test2)
    cvdd(X_iris, y_test3)
    cvdd(X_iris, y_test4)
    cvdd(X_iris, y_test5)


    # res = cvdd(X_iris, y_iris)

    # print(res)

    # X_mnist, Y_mnist = load_mnist()

    # combined_data = list(zip(X_mnist, Y_mnist))
    # np.random.shuffle(combined_data)

    # Unpack the shuffled data
    # X_mnist, Y_mnist = zip(*combined_data)

    # # Convert back to numpy arrays
    # X_mnist = np.array(X_mnist)
    # Y_mnist = np.array(Y_mnist)

    # # Select only the first 10000 samples and labels
    # X_mnist = X_mnist[:10000]
    # y_mnist = Y_mnist[:10000]

    # # print(X_mnist.shape)
    # res = cvdd(X_mnist, y_mnist)

    # print(res)


    # X_fmnist, Y_fmnist = load_fmnist()

    # # print(X_fmnist.shape)
    # # print(np.unique(Y_fmnist))

    # combined_data = list(zip(X_fmnist, Y_fmnist))
    # np.random.shuffle(combined_data)

    # # Unpack the shuffled data
    # X_fmnist, Y_fmnist = zip(*combined_data)

    # # Convert back to numpy arrays
    # X_fmnist = np.array(X_fmnist)
    # Y_fmnist = np.array(Y_fmnist)

    # # Select only the first 10000 samples and labels
    # X_fmnist = X_fmnist[:10000]
    # y_fmnist = Y_fmnist[:10000]

    # # print(X_mnist.shape)
    # res = cvdd(X_fmnist, y_fmnist)

    # print(res)

    # X_iris, y_iris = load_iris()
    # res = cvdd(X_iris, y_iris)
    # print(res)

    # clustering_kmeans = KMeans(n_clusters=3, n_init = "auto").fit(X_iris)
    # k_labels = clustering_kmeans.labels_

    # res1 = cvdd(X_iris, k_labels)
    # print(res1)


    # wine = fetch_ucirepo(id=109) 
  
    # # data (as pandas dataframes) 
    # X_wine = wine.data.features 
    # y_wine = np.array(wine.data.targets['class'])
    # from sklearn.preprocessing import LabelEncoder

    # label_encoder = LabelEncoder()

    # y_wine = label_encoder.fit_transform(y_wine)


    # cvdd(np.array(X_wine), np.array(y_wine))
    # ionosphere = fetch_ucirepo(id=52)

    # # data (as pandas dataframes)
    # X_ion = ionosphere.data.features
    # y_ion = ionosphere.data.targets

    # # metadata
    # #print(ionosphere.metadata)

    # # variable information
    # #print(ionosphere.variables)

    # #print(y_ion)

    # y_ion = label_encoder.fit_transform(y_ion['Class'])
    # print(cvdd(np.array(X_ion), np.array(y_ion))) 
    import scipy
    import sklearn
    import clustpy
    np_version = np.__version__
    scipy_version = scipy.__version__
    sklearn_version = sklearn.__version__
    clustpy_version = clustpy.__version__

    versions = {
        "numpy": np_version,
        "scipy": scipy_version,
        "scikit-learn": sklearn_version,
        "clustpy": clustpy_version
    }

    print(versions)