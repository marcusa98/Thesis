import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial import distance_matrix
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
from clustpy.data import load_iris, load_mnist, load_fmnist
from DBCV_Lena import MST_Edges
from sklearn.cluster import KMeans

# def conD(x,y):

#     return res


def build_graph(Edges): 
    # function to represent graph as dictionary of lists
    graph = defaultdict(list)
    for edge in Edges:
        u, v, w = edge
        graph[u].append((v, w))
        graph[v].append((u, w))
    #print(graph)
    return graph

def dfs(node, graph, visited, max_weights_matrix, max_weights, current_max):
    # depth first search algorithm to compute max weights for each possible path
    visited[node] = True
    # current_max was taken over from previous path, but should reset to 0 for every new immediate neighbor 
    original_max = current_max # seemingly fixes problem, but don't understand why
    for neighbor, weight in graph[node]:
        if not visited[neighbor]:
            current_max = max(original_max, weight)
            max_weights[neighbor] = max(max_weights[neighbor], current_max)

            #print((node, neighbor))
            #max_weights_matrix[int(node), int(neighbor)] = max(max_weights_matrix[int(node), int(neighbor)], current_max)

            dfs(neighbor, graph, visited, max_weights_matrix, max_weights, current_max)


def MinMaxDists(Edges):
    """
    Edges...np.array of shape n-1 x 3 with columns outgoing node, incoming node, path weight

    returns dictionary with MinMax Dists for all paths
    """
    graph = build_graph(Edges)
    nodes = np.unique(Edges[:, :2])

    #print(nodes)
    # Store all MinMax distances
    max_weights = {}
    max_weights_matrix = np.zeros((len(nodes), len(nodes)))
    # Perform DFS from each node to find maximum edge weights to other nodes
    for node in nodes:
        #print(str(node))
        #print(max_weights_matrix)
        # Initialize arrays to store maximum edge weights and visited nodes
        max_weights[str(node)] = defaultdict(int)
        visited = defaultdict(bool)
        dfs(node, graph, visited, max_weights_matrix, max_weights[str(node)], 0)
        
    #print(max_weights)
    #print(max_weights_matrix)
    
    # Compute the average of maximum edge weights
    #sum_max_weights = sum(sum(sub_dict.values()) for sub_dict in max_weights.values()) / 2 # divide by 2 because every path exists twice

    return max_weights


def cvdd(X, y, k = 5, distance = euclidean):

    n = len(y)
    stats = {}
    n_clusters = len(np.unique(y))
    seps_pairwise = np.zeros((n_clusters, n_clusters))

    # get k Nearest neighbors
    knn = NearestNeighbors(n_neighbors =  k + 1).fit(X)
    distances, _ = knn.kneighbors(X)
    #print(distances)

    # compute density estimation (Def. 2)
    stats["Den"] = np.mean(distances[:, 1:], axis=1)
    
    #print(len(stats["Den"])) # works

    # compute outlier factor (Def. 3)
    stats["fDen"] = stats["Den"] / np.max(stats["Den"])

    #print(stats["fDen"]) # works

    fRel = np.zeros((n, n))

    # set drD to euclidean distances in the beginning
    drD = distance_matrix(X, X)

    # initialize Matrix for sum of Den(x_i) and Den(x_j)
    nD = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i < j:

                # compute nD(x_i, x_j)
                nD[i, j] = stats["Den"][i] + stats["Den"][j]
                nD[j, i] = stats["Den"][j] + stats["Den"][i]

                # Relative Density (Def. 5)
                Rel_ij = stats["Den"][i] / stats["Den"][j]
                Rel_ji = stats["Den"][j] / stats["Den"][i]
                
                # mutual Density factor (Def. 6)
                fRel[i, j] = 1 - np.exp(-(Rel_ij + Rel_ji - 2))
                fRel[j, i] = 1 - np.exp(-(Rel_ji + Rel_ij - 2))

    #print(np.max(fRel))
    #print(np.min(fRel))

    relD = fRel * nD

    drD += relD
    
    #print(np.all(drD >= distance_matrix(X, X)))

    G = {
        "no_vertices": n,
        "MST_edges": np.zeros((n - 1, 3)),
        "MST_degrees": np.zeros((n), dtype=int),
        "MST_parent": np.zeros((n), dtype=int),
    }


    #print(drD)

    # compute MST using Lena's implementation of Prim algorithm
    Edges_drD, _ = MST_Edges(G, 0, drD)   

    #print(Edges_drD)

    conD = MinMaxDists(Edges_drD)

    #print(conD)

    # filling up DD Matrix (Def. 10)
    DD = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i < j:
                #pass
                #print(conD[str(float(i))][float(j)] == conD[str(float(j))][float(i)])
                DD[i,j] = ((stats["fDen"][i] * stats["fDen"][j]) ** 0.5) * conD[str(float(i))][float(j)]
                DD[j,i] = ((stats["fDen"][i] * stats["fDen"][j]) ** 0.5) * conD[str(float(j))][float(i)]

    #print(DD)

    # Compute separations between one cluster and all others (Def. 11)
    seps = []

    for i in range(n_clusters):
            seps.append(np.min(DD[np.ix_(np.where(y == i)[0], np.where(y != i)[0])]))

    #print(seps)

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
        inter_dists = distance_matrix(X[np.where(y == cluster)[0],:], X[np.where(y == cluster)[0],:])

        #print(inter_dists)

        # compute MST using Lena's implementation of Prim algorithm
        Edges_pD, _ = MST_Edges(G, 0, inter_dists)


        #print(Edges_pD)

        pD = MinMaxDists(Edges_pD)

        # flattening pD Dictionary
        flattened_data = [value for sub_dict in pD.values() for value in sub_dict.values()]

        Mean_Ci = np.mean(flattened_data)
        Std_Ci = np.std(flattened_data)

        # print(Mean_Ci)
        # print(Std_Ci)

        # compute compactness (Def. 12)
        #com_Ci = (1/n_paths) * Std_Ci * Mean_Ci
        com_Ci = (1 / cluster_size) * Std_Ci * Mean_Ci

        #print(com_Ci)
        coms.append(com_Ci)


    #print(seps)
    #print(coms)

    # compute CVDD (Def. 13)
    res = sum(seps) / sum(coms)

    #print(res)
    return res


X_iris, y_iris = load_iris()
res = cvdd(X_iris, y_iris)
print(res)

clustering_kmeans = KMeans(n_clusters=3, n_init = "auto").fit(X_iris)
k_labels = clustering_kmeans.labels_

res1 = cvdd(X_iris, k_labels)
print(res1)


if __name__ == "__main__":
    from numpy.random import MT19937
    from numpy.random import RandomState, SeedSequence
    rs = RandomState(MT19937(SeedSequence(123456789)))
    # n = 10
    
    # G = {
    #     "no_vertices": n,
    #     "MST_edges": np.zeros((n - 1, 3)),
    #     "MST_degrees": np.zeros((n), dtype=int),
    #     "MST_parent": np.zeros((n), dtype=int),
    # }

    # data = np.random.rand(n, 3)

    # drD = distance_matrix(data, data)

    # # compute MST using Lena's implementation of Prim algorithm
    # Edges_drD, _ = MST_Edges(G, 0, drD)   

    # print(Edges_drD)

    # graph = build_graph(Edges_drD)

    # print(graph)

    # nodes = np.unique(Edges_drD[:, :2])

    # print(nodes)



    Edges = np.array([[0, 1, 0.43], [1, 2, 0.6], [2, 3, 0.5]])

    MinMaxDists(Edges)

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