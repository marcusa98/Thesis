import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, squareform
from DBCV_Lena import MST_Edges


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
    n_clusters = len(np.unique(y))
    seps = np.zeros((n_clusters, n_clusters))
    connects = np.zeros((n_clusters, n_clusters))


    for cluster in np.unique(y):
        #eps.append() #compute epsilon for cluster
        corepoints_cluster = []

        # compute all pairwise distances inside a cluster
        #inter_dists = distance_matrix(X[np.where(y == cluster)[0],:], X[np.where(y == cluster)[0],:])
        inter_dists = squareform(pdist(X[np.where(y == cluster)[0],:]))


        # compute eps for cluster according to Definition 4.6
        # set first minPts*2 - 1 closest distances to np.inf
        # this is supposed to be faster than sorting the whole matrix and selecting the respective column
        temp = inter_dists.copy()
        for _ in range(minPts*2): # minPts*2, first minima is 0 because its point itself and then minPts*2 - 1 closest points
            min_inds = np.argmin(temp, axis = 1)
            rows = np.arange(temp.shape[0])
            temp[rows, min_inds] = np.inf
            
        # set eps to median of distance to minPts*2-th closest point
        eps["eps " + str(cluster)] = np.median(np.min(temp, axis = 1))
        

        #print(eps)
        
        # store all corepoints in a cluster to compute separation
        for indx, point_dists in enumerate(inter_dists):
            if sum(1 for value in point_dists if value <= eps["eps " + str(cluster)]) >= (minPts + 1):  # because a point is always in it's own eps-neighborhood
                corepoints_cluster.append(list(X[np.where(y == cluster)[0],:][indx,:])) # add point to corepoints_cluster

        # add corepoints of every cluster to corepoints_all
        corepoints_all["cluster " + str(cluster)] = np.array(corepoints_cluster)
        cluster_sizes["cluster " + str(cluster)] = len(np.where(y == cluster)[0])


        # get corepoint indices
        corepoint_inds = np.sum(inter_dists <= eps["eps " + str(cluster)], axis=1) >= (minPts + 1) # +1 because distance to itself is 0

        #print(corepoint_inds)

        # extract the distancematrix for corepoints only
        inter_dists_corepoints = inter_dists[corepoint_inds][:, corepoint_inds]

        # Print the matrices
        #print(inter_dists.shape)
        #print(inter_dists_corepoints)
        #print(len(corepoints_cluster))

        if len(corepoints_cluster) < 2:
            raise ValueError("At least 2 core points are required. Try to increase minPts")

        # create empty graph on corepoints
        G = {
            "no_vertices": len(corepoints_cluster),
            "MST_edges": np.zeros((len(corepoints_cluster) - 1, 3)),
            "MST_degrees": np.zeros((len(corepoints_cluster)), dtype=int),
            "MST_parent": np.zeros((len(corepoints_cluster)), dtype=int),
        }

        # compute MST using Lena's implementation of Prim algorithm
        Edges, Degrees = MST_Edges(G, 0, inter_dists_corepoints)

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

    #return corepoints_all



if __name__ == "__main__":
    #import umap.umap_ as umap
    from clustpy.data import load_iris, load_mnist, load_fmnist
    from sklearn import datasets
    # X = np.array([[1,2,3,4],[5,6,7,8]])
    # y = np.array([0,3])

    #print(len(np.unique(y)))
    #dcsi(X,y)

    # X_iris, y_iris = load_iris()
    
    # res = dcsi(X_iris, y_iris)

    # print(res)

    X_mnist, Y_mnist = load_mnist()

    combined_data = list(zip(X_mnist, Y_mnist))
    np.random.shuffle(combined_data)

    # Unpack the shuffled data
    X_mnist, Y_mnist = zip(*combined_data)

    # Convert back to numpy arrays
    X_mnist = np.array(X_mnist)
    Y_mnist = np.array(Y_mnist)

    # Select only the first 10000 samples and labels
    X_mnist = X_mnist[:10000]
    y_mnist = Y_mnist[:10000]

    # standardize as matrix
    X_mnist_norm = (X_mnist - np.mean(X_mnist)) / np.std(X_mnist)

    # print(X_mnist.shape)
    res = dcsi(X_mnist, y_mnist)

    print(res) # 0.451

    X_fmnist, Y_fmnist = load_fmnist()

    # print(X_fmnist.shape)
    # print(np.unique(Y_fmnist))

    combined_data = list(zip(X_fmnist, Y_fmnist))
    np.random.shuffle(combined_data)

    # Unpack the shuffled data
    X_fmnist, Y_fmnist = zip(*combined_data)

    # Convert back to numpy arrays
    X_fmnist = np.array(X_fmnist)
    Y_fmnist = np.array(Y_fmnist)

    # Select only the first 10000 samples and labels
    X_fmnist = X_fmnist[:10000]
    y_fmnist = Y_fmnist[:10000]

    # standardize as matrix
    X_fmnist_norm = (X_fmnist - np.mean(X_fmnist)) / np.std(X_fmnist)

    # print(X_mnist.shape)
    res = dcsi(X_fmnist, y_fmnist)

    print(res) # 0.403

    # print(np.unique(y_mnist))
    # print(np.unique(y_fmnist))

    # create UMAP embedding for Mnist
    fit = umap.UMAP(
        n_neighbors=10, 
        min_dist=0.1,
        n_components=3,
        metric="euclidean"
    )

    umap_mnist = fit.fit_transform(X_mnist_norm)

    res = dcsi(umap_mnist, y_mnist)

    print(res) # 0.776

    # create UMAP embedding for FMnist-10
    fit = umap.UMAP(
        n_neighbors=10, #maybe 15
        min_dist=0.1,
        n_components=3,
        metric="euclidean"
    )

    umap_fmnist = fit.fit_transform(X_fmnist_norm)

    res = dcsi(umap_fmnist, y_fmnist)

    print(res) # 0.541




    moons = datasets.make_moons(noise = 0.05, n_samples = 150)
    X,y = moons[0], moons[1]

    dcsi(X,y) #why is minPts = 60 possible?

    #########

    X = np.random.rand(7,10)
    y = np.random.rand(3,10)
    print(distance_matrix(X,y))


    stacked_matrix = np.vstack((X, y))

    full_distance_matrix = squareform(pdist(stacked_matrix))

    # Extract the relevant submatrix
    # matrix_i has shape (m, n) and matrix_j has shape (p, n)
    m = X.shape[0]
    p = y.shape[0]

    # The distances between points in matrix_i and matrix_j are in the submatrix
    distances_between_sets = full_distance_matrix[:m, m:m+p]

    print(distances_between_sets)