import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import pairwise_distances
from numba import jit

def dunn_index(X, labels):
    """
    Compute the Dunn Index of a clustering solution.

    Parameters:
        X (array-like): Dataset.
        labels (array-like): Cluster labels for each data point.

    Returns:
        float: Dunn Index of the clustering solution.
    """
    # Calculate pairwise distances between points in the dataset
    #distances = squareform(pdist(X))
    distances = pairwise_distances(X, X)

    # Calculate separability (minimum distance between any two points between any two clusters)
    min_inter_cluster_distance = np.min(distances[labels != labels[:, None]])

    # Calculate compactness (maximum distance between any two points of any cluster)
    max_intra_cluster_distance = np.max(distances[labels == labels[:, None]])

    # Compute Dunn Index
    dunn_index = min_inter_cluster_distance / max_intra_cluster_distance

    return dunn_index


if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    from clustpy.data import load_iris, load_mnist
    from ucimlrepo import fetch_ucirepo 
    from sklearn.preprocessing import LabelEncoder
    from time import time
    from tqdm import trange
    from scipy.sparse import csr_matrix
    from sklearn.metrics.pairwise import pairwise_distances
    
    # moons = datasets.make_moons(noise = 0.05, n_samples = 1500)
    # X,y = moons[0], moons[1]

    # clustering_kmeans = KMeans(n_clusters=2, n_init = "auto").fit(X)
    # k_labels = clustering_kmeans.labels_

    # res = dunn_index(X, y)
    # print(res)


    # plt.subplot(1,2,1)
    # plt.scatter(X[:,0], X[:,1], c=y)
    # plt.title("Clustering K-Means")
    # plt.axis('equal')
    # plt.tight_layout()
    # plt.show()

    mnist = load_mnist()
    X, y = mnist['data'], mnist['target']

    # X = np.array(X)[:5000]
    # y = np.array(y)[:5000]

    # start = time()
    # print(dunn_index(X, y)) # mnist[:10000]: dunn_index = 0.21147
    # stop = time()
    # print(stop-start) # csr_matrix: 15.39 sec, without csr_matrix: 26.62 sec


    # X = np.array(X)[:10000]
    # y = np.array(y)[:10000]

    # start = time()
    # print(dunn_index(X, y)) # mnist[:5000]: dunn_index = 0.25569
    # stop = time()
    # print(stop-start) #  squareform(pdist(X)): 4.46 sec/ pairwise_distance(X,X): 0.457


    X = np.array(X)[:10000]
    y = np.array(y)[:10000]

    start = time()
    print(dunn_index(X, y)) # mnist[:10000]: dunn_index = 0.21147
    stop = time()
    print(stop-start) #  squareform(pdist(X)): 22.059 sec/ pairwise_distance(X,X): 1.953 sec

