import numpy as np
from scipy.spatial import distance_matrix

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
    distances = distance_matrix(X, X)

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
    moons = datasets.make_moons(noise = 0.05, n_samples = 150)
    X,y = moons[0], moons[1]

    clustering_kmeans = KMeans(n_clusters=2, n_init = "auto").fit(X)
    k_labels = clustering_kmeans.labels_

    res = dunn_index(X, y)
    print(res)


    plt.subplot(1,2,1)
    plt.scatter(X[:,0], X[:,1], c=y)
    plt.title("Clustering K-Means")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()