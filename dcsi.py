import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial import distance_matrix
from clustpy.data import load_iris
from DBCV_Lena import MST_Edges


def separation(x,y):
    """
    x...corepoints of cluster i
    y...corepoints of cluster j
    """
    dists_ij = distance_matrix(x, y)
    sep = np.min(dists_ij)

    return sep


def dcsi(X, y, eps = 0.6, minPts = 5, distance = euclidean):
    """
    X... np.array n x p
    y... data partition
    eps... defines the epsilon neighborhood
    minPts... number of points contained inside the epsilon neighborhood of a point, for the point to be considered a corepoint

    returns: value between 0 and 1
    """
    eps = {}
    corepoints_all = {}
    cluster_sizes = {}
    connectedness = {}
    n_clusters = len(np.unique(y))
    seps = np.zeros((n_clusters, n_clusters))
    connects = np.zeros((n_clusters, n_clusters))


    for cluster in np.unique(y):
        #eps.append() #compute epsilon for cluster
        corepoints_cluster = []

        # compute all pairwise distances inside a cluster
        inter_dists = distance_matrix(X[np.where(y == cluster)[0],:], X[np.where(y == cluster)[0],:])

        # compute eps for cluster according to Definition 4.6
        # set minimum to np.inf 
        temp = inter_dists.copy()
        for _ in range(minPts*2): 
            min_inds = np.argmin(temp, axis = 1)
            rows = np.arange(temp.shape[0])
            temp[rows, min_inds] = np.inf
            
        #print(np.min(temp, axis = 1))
        eps["eps " + str(cluster)] = np.median(np.min(temp, axis = 1))

        # store all corepoints in a cluster
        for point in inter_dists:
            if sum(1 for value in point if value <= eps["eps " + str(cluster)]) >= (minPts + 1):  # because a point is always in it's own eps-neighborhood
                corepoints_cluster.append(point)

        # add corepoints of every cluster to corepoints_all
        corepoints_all["cluster " + str(cluster)] = corepoints_cluster
        cluster_sizes["cluster " + str(cluster)] = len(np.where(y == cluster)[0])


        # get corepoint indices
        corepoint_inds = np.sum(inter_dists <= eps["eps " + str(cluster)], axis=1) >= (minPts + 1) # +1 because distance to itself is 0

        # Use boolean indexing to extract the distances for core points only
        inter_dists_corepoints = inter_dists[corepoint_inds][:, corepoint_inds]

        # Print the matrices
        #print(inter_dists.shape)
        #print(inter_dists_corepoints)
        #print(len(corepoints_cluster))

        # create empty graph
        G = {
            "no_vertices": len(corepoints_cluster),
            "MST_edges": np.zeros((len(corepoints_cluster) - 1, 3)),
            "MST_degrees": np.zeros((len(corepoints_cluster)), dtype=int),
            "MST_parent": np.zeros((len(corepoints_cluster)), dtype=int),
        }

        # compute MST using Lena's implementation of Prim algorithm
        Edges, Degrees = MST_Edges(G, 0, inter_dists_corepoints)

        # print(Edges)
        # print(Degrees)
        connectedness["cluster " + str(cluster)] = np.max(Edges[:,2])

        # print(connectedness)
    keys_1 = list(corepoints_all.keys())
    keys_2 = list(connectedness.keys())
    

    for i in range(n_clusters):
        for j in range(n_clusters):
            if i != j:  # compute pairwise separations and connectedness
                seps[i, j] = separation(corepoints_all[keys_1[i]], corepoints_all[keys_1[j]])
                connects[i, j] = max(connectedness[keys_2[i]], connectedness[keys_2[j]])

    # print(seps)        
    # print(connects)


    if len(np.unique(y)) == 2:
        #print("2 classes only")
        conn = connects[1, 1]
        sep = seps[1, 1]
        q = sep / conn
        return q / (1 + q)

    else:
        #print("more than 2 classes")
        dcsis = []
        for i in range(n_clusters):
            for j in range(n_clusters):
                if i < j:  # compute pairwise separations and connectedness
                    conn = connects[i, j]
                    sep = seps[i, j]
                    q = sep / conn
                    dcsis.append((q / (1 + q))) 


        return np.mean(dcsis)

    #return corepoints_all



if __name__ == "__main__":

    X = np.array([[1,2,3,4],[5,6,7,8]])
    y = np.array([0,3])



    #print(len(np.unique(y)))
    #dcsi(X,y)

    X_iris, y_iris = load_iris()
    
    res = dcsi(X_iris, y_iris)

    print(res)

    # start = []
    # start.append([1,2,3,2])
    # start.append([3,3,3,3])
    # print(start)



