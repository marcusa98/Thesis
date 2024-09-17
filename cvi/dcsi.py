import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, squareform
from mst import MST_Edges

def separation(x,y):
    """
    x...corepoints of cluster i
    y...corepoints of cluster j
    """

    stacked_matrix = np.vstack((x, y))
    full_distance_matrix = squareform(pdist(stacked_matrix))

    # Extract the relevant submatrix
    # matrix_i has shape (m, n) and matrix_j has shape (p, n)
    m = x.shape[0]
    p = y.shape[0]

    # The distances between points in matrix_i and matrix_j are in the submatrix
    dists_ij = full_distance_matrix[:m, m:m+p]
    #dists_ij = distance_matrix(x, y)

    sep = np.min(dists_ij)

    return sep


def dcsi(X, y, eps = 0.6, minPts = 5, distance = euclidean):
    """
    Computes DCSI of a clustering solution.

    Parameters:
        X (array-like): Dataset.
        y (array-like): Cluster labels for each data point.
        eps (list): defines epsilon neighborhood of each cluster
        minPts (integer): number of points contained inside the epsilon neighborhood of a point, for the point to be considered a corepoint
    Returns:
        float: DCSI (value between 0 and 1) of the clustering solution.
    """
    eps = {}
    corepoints_all = {}
    cluster_sizes = {}
    connectedness = {}

    # remove noise from labels
    labels = y
    y = y[labels != -1]

    # remove noise from X
    X = X[labels != -1, :]

    n_clusters = len(np.unique(y))
    seps = np.zeros((n_clusters, n_clusters))
    connects = np.zeros((n_clusters, n_clusters))


    for cluster in np.unique(y):
        #eps.append() #compute epsilon for cluster
        corepoints_cluster = []

        # compute all pairwise distances inside a cluster
        #intra_dists = distance_matrix(X[np.where(y == cluster)[0],:], X[np.where(y == cluster)[0],:])
        intra_dists = squareform(pdist(X[np.where(y == cluster)[0],:]))


        # compute eps for cluster according to Definition 4.6
        # set first minPts*2 - 1 closest distances to np.inf
        # this is supposed to be faster than sorting the whole matrix and selecting the respective column
        temp = intra_dists.copy()
        for _ in range(minPts*2): # minPts*2, first minima is 0 because its point itself and then minPts*2 - 1 closest points
            min_inds = np.argmin(temp, axis = 1)
            rows = np.arange(temp.shape[0])
            temp[rows, min_inds] = np.inf
            
        # set eps to median of distance to minPts*2-th closest point
        eps["eps " + str(cluster)] = np.median(np.min(temp, axis = 1))
                
        # store all corepoints in a cluster to compute separation
        for indx, point_dists in enumerate(intra_dists):
            if sum(1 for value in point_dists if value <= eps["eps " + str(cluster)]) >= (minPts + 1):  # because a point is always in it's own eps-neighborhood
                corepoints_cluster.append(list(X[np.where(y == cluster)[0],:][indx,:])) # add point to corepoints_cluster

        # add corepoints of every cluster to corepoints_all
        corepoints_all["cluster " + str(cluster)] = np.array(corepoints_cluster)
        cluster_sizes["cluster " + str(cluster)] = len(np.where(y == cluster)[0])


        # get corepoint indices
        corepoint_inds = np.sum(intra_dists <= eps["eps " + str(cluster)], axis=1) >= (minPts + 1) # +1 because distance to itself is 0

        #print(corepoint_inds)

        # extract the distancematrix for corepoints only
        intra_dists_corepoints = intra_dists[corepoint_inds][:, corepoint_inds]


        if len(corepoints_cluster) < 2:
            raise ValueError("At least 2 core points are required. Try to decrease minPts")

        # create empty graph on corepoints
        G = {
            "no_vertices": len(corepoints_cluster),
            "MST_edges": np.zeros((len(corepoints_cluster) - 1, 3)),
            "MST_degrees": np.zeros((len(corepoints_cluster)), dtype=int),
            "MST_parent": np.zeros((len(corepoints_cluster)), dtype=int),
        }

        # compute MST using Lena's implementation of Prim algorithm
        Edges, Degrees = MST_Edges(G, 0, intra_dists_corepoints)

        connectedness["cluster " + str(cluster)] = np.max(Edges[:,2])

        # print(connectedness)
    keys_1 = list(corepoints_all.keys())
    keys_2 = list(connectedness.keys())
    

    for i in range(n_clusters):
        for j in range(n_clusters):
            if i != j:  # compute pairwise separations and connectedness
                seps[i, j] = separation(corepoints_all[keys_1[i]], corepoints_all[keys_1[j]])
                connects[i, j] = max(connectedness[keys_2[i]], connectedness[keys_2[j]])


    if len(np.unique(y)) == 2:
        #print("2 classes only")
        conn = connects[0, 1]
        sep = seps[0, 1]
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



if __name__=="__main__":
    # from clustpy.data import load_har, load_letterrecognition, load_htru2, load_mice_protein, load_pendigits,\
    # load_coil20, load_coil100, load_cmu_faces, load_optdigits, load_usps, load_mnist, load_fmnist, load_kmnist ,load_video_keck_gesture, load_video_weizmann
    # from ucimlrepo import fetch_ucirepo 
    # from sklearn.preprocessing import LabelEncoder
    # # create Label encoder 
    # label_encoder = LabelEncoder()

    # iris = fetch_ucirepo(id=53)

    # # data (as pandas dataframes)
    # X_iris = np.array(iris.data.features)
    # y_iris = iris.data.targets

    # y_iris = np.array(label_encoder.fit_transform(y_iris['class']))

    # # # print(X_iris)
    # # # print(y_iris)
    # print(dcsi(X_iris, y_iris))