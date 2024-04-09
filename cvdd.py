import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial import distance_matrix
from sklearn.neighbors import NearestNeighbors
from clustpy.data import load_iris, load_mnist, load_fmnist
from DBCV_Lena import MST_Edges


def conD(x,y):

    return res


def cvdd(X, y, k = 5, distance = euclidean):

    n = len(y)
    stats = {}

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
    




if __name__ == "__main__":

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