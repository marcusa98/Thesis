import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial import distance_matrix
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
from clustpy.data import load_iris, load_mnist, load_fmnist
from DBCV_Lena import MST_Edges


def conD(x,y):

    return res


def build_graph(Edges): 
    # function to represent graph as dictionary of lists
    graph = defaultdict(list)
    for edge in Edges:
        u, v, w = edge
        graph[u].append((v, w))
        graph[v].append((u, w))
    print(graph)
    return graph

def dfs(node, graph, visited, max_weights, current_max):
    # depth first search algorithm to compute max weights for each possible path
    visited[node] = True
    # current_max was taken over from previous path, but should reset to 0 for every new immediate neighbor 
    original_max = current_max # seemingly fixes problem, but don't understand why
    for neighbor, weight in graph[node]:
        if not visited[neighbor]:
            current_max = max(original_max, weight)
            max_weights[neighbor] = max(max_weights[neighbor], current_max)
            dfs(neighbor, graph, visited, max_weights, current_max)


def sum_max_edge_weights(Edges):
    graph = build_graph(Edges)
    nodes = np.unique(Edges[:, :2])

    # Store all MinMax distances
    max_weights = {}

    # Perform DFS from each node to find maximum edge weights to other nodes
    for node in nodes:
        print(str(node))
        # Initialize arrays to store maximum edge weights and visited nodes
        max_weights[str(node)] = defaultdict(int)
        visited = defaultdict(bool)
        dfs(node, graph, visited, max_weights[str(node)], 0)
        
    print(max_weights)

    # Compute the average of maximum edge weights
    sum_max_weights = sum(sum(sub_dict.values()) for sub_dict in max_weights.values()) / 2 # divide by 2 because every path exists twice

    return sum_max_weights


def cvdd(X, y, k = 5, distance = euclidean):

    n = len(y)
    stats = {}
    cluster_sizes = {}


    # get k Nearest neighbors
    knn = NearestNeighbors(n_neighbors =  k + 1).fit(X)
    distances, _ = knn.kneighbors(X)

    # compute density estimation (Def. 2)
    stats["Den"] = np.mean(distances[:, 1:], axis=1)
    

    # compute outlier factor (Def. 3)
    stats["fDen"] = stats["Den"] / np.max(stats["Den"])

    #print(stats)


    # filling up DD Matrix (Def. 10)
    DD = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                pass
                # DD[i,j] = ((stats["fDen"][i] * stats["fDen"][j]) ** 0.5) * conD[i,j]
                # DD[j,i] = ((stats["fDen"][i] * stats["fDen"][j]) ** 0.5) * conD[i,j]


    # filling up Rel Matrix (Def. 5)
    #Rel = np.ones((n, n))

    fRel = np.zeros((n, n))

    # set drD to euclidean distances in the beginning
    drD = distance_matrix(X, X)

    # initialize Matrix for sum of Den(x_i) and Den(x_j)
    Dens_added = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i < j:

                # compute nD(x_i, x_j)
                Dens_added[i, j] = stats["Den"][i] + stats["Den"][j]
                Dens_added[j, i] = stats["Den"][j] + stats["Den"][i]

                # Relative Density (Def. 5)
                Rel_ij = stats["Den"][i] / stats["Den"][j]
                Rel_ji = stats["Den"][j] / stats["Den"][i]
                
                # mutual Density factor (Def. 6)
                fRel[i, j] = 1 - np.exp(1)**-(Rel_ij + Rel_ji - 2)
                fRel[j, i] = 1 - np.exp(1)**-(Rel_ji + Rel_ij - 2)

    relD = fRel * Dens_added

    drD += relD
    
    print(drD)
    

    for cluster in np.unique(y):

        cluster_sizes["Cluster" + str(cluster)] = len(np.where(y == cluster)[0])


        G = {
            "no_vertices": len(np.where(y == cluster)[0]),
            "MST_edges": np.zeros((len(np.where(y == cluster)[0]) - 1, 3)),
            "MST_degrees": np.zeros((len(np.where(y == cluster)[0])), dtype=int),
            "MST_parent": np.zeros((len(np.where(y == cluster)[0])), dtype=int),
        }

        # compute all pairwise distances inside a cluster
        inter_dists = distance_matrix(X[np.where(y == cluster)[0],:], X[np.where(y == cluster)[0],:])

        # compute MST using Lena's implementation of Prim algorithm
        Edges, Degrees = MST_Edges(G, 0, inter_dists)

        print(Edges)


if __name__ == "__main__":

    
    # Edges = np.array([[1, 2, 0.43], [2, 3, 0.6], [3, 4, 0.5]])

    # sum_max_edge_weights(Edges)

    # X = np.array([[1,2,3,4],[5,6,7,8]])
    # y = np.array([0,3])

    #print(len(np.unique(y)))
    #dcsi(X,y)

    X_iris, y_iris = load_iris()
    
    cvdd(X_iris, y_iris)

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