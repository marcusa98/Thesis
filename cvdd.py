import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial import distance_matrix
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
from clustpy.data import load_iris, load_mnist, load_fmnist
from DBCV_Lena import MST_Edges


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

    returns dicitonary with MinMax Dists for all paths
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

    # compute density estimation (Def. 2)
    stats["Den"] = np.mean(distances[:, 1:], axis=1)
    

    # compute outlier factor (Def. 3)
    stats["fDen"] = stats["Den"] / np.max(stats["Den"])

    #print(stats)

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
    
    #print(drD)

    G = {
        "no_vertices": n,
        "MST_edges": np.zeros((n - 1, 3)),
        "MST_degrees": np.zeros((n), dtype=int),
        "MST_parent": np.zeros((n), dtype=int),
    }

    # compute MST using Lena's implementation of Prim algorithm
    Edges_drD, _ = MST_Edges(G, 0, drD)   

    conD = MinMaxDists(Edges_drD)


    #print(list(conD.values())[0])
    # print(list(conD.keys())[0] == str(float(0)))

    # print(conD[str(float(0))])
    # print(conD[str(float(0))][float(12)])
    # print(conD[int(float(0))][int(float(2))])
    # print("conD")

    # filling up DD Matrix (Def. 10)
    DD = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                pass
                DD[i,j] = ((stats["fDen"][i] * stats["fDen"][j]) ** 0.5) * conD[str(float(i))][float(j)]
                DD[j,i] = ((stats["fDen"][i] * stats["fDen"][j]) ** 0.5) * conD[str(float(j))][float(i)]


    # Compute pairwise Separation between all Clusters (Def. 11)
    
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i < j:  
                seps_pairwise[i, j] = min(DD[np.where(y == i)[0], np.where(y == j)[0]])
                seps_pairwise[j, i] = min(DD[np.where(y == i)[0], np.where(y == j)[0]])

    seps_pairwise[np.where(seps_pairwise == 0)] = np.inf

    #print(seps_pairwise)
    # get minimal separation between one cluster and others
    seps = np.min(seps_pairwise, axis= 0)

    #print(seps)


    coms = []

    for cluster in np.unique(y):

        cluster_size = len(np.where(y == cluster)[0])


        G = {
            "no_vertices": cluster_size,
            "MST_edges": np.zeros((cluster_size - 1, 3)),
            "MST_degrees": np.zeros((cluster_size), dtype=int),
            "MST_parent": np.zeros((cluster_size), dtype=int),
        }

        # compute all pairwise distances inside a cluster
        inter_dists = distance_matrix(X[np.where(y == cluster)[0],:], X[np.where(y == cluster)[0],:])

        # compute MST using Lena's implementation of Prim algorithm
        Edges_pD, _ = MST_Edges(G, 0, inter_dists)

        pD = MinMaxDists(Edges_pD)

        # compute sum of pD inside the cluster
        sum_pD = sum(sum(sub_dict.values()) for sub_dict in pD.values()) / 2 # divide by 2 because every path exists twice
        #print(sum_pD)

        Mean_Ci = 1 / cluster_size * sum_pD 

        #print(Mean_Ci)

        Std_Ci = ((sum_pD - Mean_Ci * (cluster_size * (cluster_size - 1) / 2)) / (1 / cluster_size - 1))**0.5

        #print(Std_Ci)
        # compute compactness (Def. 12)
        com_Ci = (Std_Ci * Mean_Ci) / cluster_size 

        print()
        #print(com_Ci)
        coms.append(com_Ci)

    coms = np.array(coms)

    # compute CVDD (Def. 13)
    res = np.sum(seps) / np.sum(coms)

    #print(res)
    return res

res = cvdd(X_iris, y_iris)
print(res)







if __name__ == "__main__":

    
    Edges = np.array([[0, 1, 0.43], [1, 2, 0.6], [2, 3, 0.5]])

    MinMaxDists(Edges)

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