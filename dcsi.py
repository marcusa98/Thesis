import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial import distance_matrix
from clustpy.data import load_iris



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
    minPts... number of points contained inside the epsilon neighborhood of a point to be considered a corepoint

    returns value between 0 and 1
    """
    #eps = []
    minPts += 1 # because a point is always in it's own eps-neighborhood
    corepoints_all = {}
    cluster_sizes = {}
    n_clusters = len(np.unique(y))
    separations = np.zeros((n_clusters, n_clusters))

    for cluster in np.unique(y):
        #eps.append() #compute epsilon for cluster
        corepoints_cluster = []

        # compute all pairwise distances inside a cluster
        inter_dists = distance_matrix(X[np.where(y == cluster)[0],:], X[np.where(y == cluster)[0],:])

        # compute eps for cluster according to Definition 4.6
        # set minimum to np.inf 
        temp = inter_dists
        for i in range(minPts*2 + 1): # + 1 because first np.min() is point itself
            # print(temp)
            min_inds = np.argmin(temp, axis = 1)
            # print(min_inds)
            rows = np.arange(temp.shape[0])
            temp[rows, min_inds] = np.inf
            # print(temp)
            
        #print(np.min(temp, axis = 1))
        eps = np.median(np.min(temp, axis = 1))

        print("cluster " + str(cluster) + ":" + str(eps))

        # store all corepoints in a cluster
        for point in inter_dists:
            if sum(1 for value in point if value <= eps) >= minPts:
                corepoints_cluster.append(point)

        # add corepoints of every cluster to corepoints_all
        corepoints_all["cluster " + str(cluster)] = corepoints_cluster
        cluster_sizes["cluster " + str(cluster)] = len(np.where(y == cluster)[0])


    keys = list(corepoints_all.keys())

    for i in range(n_clusters):
        for j in range(n_clusters):
            if i != j:  # Avoid computing distance between the same list
                separations[i, j] = separation(corepoints_all[keys[i]], corepoints_all[keys[j]])

    #print(separations)



    if len(np.unique(y)) == 2:
        print("2 classes only")
        #return dcsi
    else:
        print("more than 2 classes")
        #return dcsi

    #return corepoints_all



if __name__ == "__main__":

    X = np.array([[1,2,3,4],[5,6,7,8]])
    y = np.array([0,3])

    #print(len(np.unique(y)))
    #dcsi(X,y)

    #print(euclidean(X))

    X_iris, y_iris = load_iris()

    res = dcsi(X_iris, y_iris)

    print(res)

    
    # start = []
    # start.append([1,2,3,2])
    # start.append([3,3,3,3])
    # print(start)



