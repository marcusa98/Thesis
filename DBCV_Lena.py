import numpy as np
from scipy.spatial.distance import pdist, squareform


#########################################################################################
#   Differences between this and the hdbscan implementation:                            #
#       - the authors use Prim, hdbscan works with Kruskal                              #
#       - instead of just taking the pairwise distance (hdbscan package)                #
#           they use the squared version                                                #
#   Translated from MATLAB: https://github.com/pajaskowiak/dbcv/tree/main               #
#   Paper: https://www.dbs.ifi.lmu.de/~zimek/publications/SDM2014/DBCV.pdf              #
#                                                                                       #
#########################################################################################
def matrix_mutual_reachability_distance(MinPts, G_edges_weights, d):
    """
    Calculate graph based on mutual reachability distances.
    Returns all-points-core-distance between the datapoints and the graph.

    d_mreach(o_i,o_j) = max{a_{pts}coredist(o_i),a_{pts}coredist(o_j),d(o_i,o_j)}

    :param MinPts: number of points in Cluster
    :param G_edges_weights: distance matrix for all points included in cluster i
    :param d: dimension of dataset
    :return: d_ucore, G_edges_weights
    """
    # number of datapoint in i and j
    No = G_edges_weights.shape[0]
    # K-NN needed for all-points-core-distance (Definition 1, equation 3.1)
    K_NN_Dist = np.power(G_edges_weights, -1 * d)
    K_NN_Dist[K_NN_Dist == np.inf] = 0

    # original
    # all-points-core distance (Definition 1, equation 3.1)
    d_ucore = np.sum(K_NN_Dist, axis=0)
    d_ucore = d_ucore / (No - 1)
    d_ucore = np.power(1.0 / d_ucore, 1 / (1 * d))
    d_ucore[d_ucore == np.inf] = 0

    # for all points
    for i in range(No):
        # we look at all points in the cluster
        for j in range(MinPts):
            # Mutual reachability distance (Definition 2)
            G_edges_weights[i, j] = max(d_ucore[i], d_ucore[j], G_edges_weights[i, j])
            G_edges_weights[j, i] = G_edges_weights[i, j]

    return d_ucore, G_edges_weights


def MST_Edges(G, start, G_edges_weights):
    """
    Calculate minimal spanning tree.
    Returning edges of MST and degrees for each node.

    :param G: dict including information
    :param start: starting point for Prim
    :param G_edges_weights: matrix including edge weights
    :return: Edg, Degr
    """
    # Prims algorithm
    # distance array
    d = []
    # array storing if already visited
    intree = []
    # initialize
    for i in range(G["no_vertices"]):
        # add first no node is already visited
        intree.append(0)
        # the distance to all vertices is inf
        d.append(np.inf)
        # Parents are set to themselves
        G["MST_parent"][i] = int(i)
    # starting at start
    d[start] = 0
    # pointer
    v = start
    counter = -1
    # count until we connected all nodes
    while counter < G["no_vertices"] - 2:
        # we add v to the 'visited'
        intree[v] = 1
        dist = np.inf
        # for every node
        for w in range(G["no_vertices"]):
            # if the node is not already in the visited elements and is not the same as the one we want to check
            if (w != v) & (intree[w] == 0):
                # we look at the distance
                weight = G_edges_weights[v, w]
                # if the distance is smaller than the distance that connects w currently we update
                if d[w] > weight:
                    d[w] = weight
                    G["MST_parent"][w] = int(v)
                # if the distance is smaller than current dist we update dist
                if dist > d[w]:
                    dist = d[w]
                    next_v = w
        # update counter for dict
        counter = counter + 1
        # the outgoing node is the parent
        outgoing = G["MST_parent"][next_v]
        # incoming is the current vertex
        incoming = next_v
        # add a new edge
        G["MST_edges"][counter, :] = [
            outgoing,
            incoming,
            G_edges_weights[outgoing, incoming],
        ]
        # update degrees for outgoing and incoming
        G["MST_degrees"][G["MST_parent"][next_v]] = (
                G["MST_degrees"][G["MST_parent"][next_v]] + 1
        )
        G["MST_degrees"][next_v] = G["MST_degrees"][next_v] + 1
        # set next vertex
        v = next_v
    # extract Edges and degrees from the dict
    Edg = G["MST_edges"]
    Degr = G["MST_degrees"]
    return Edg, Degr


def dbcv(data, partition):
    """
    Calculate the Density Based Clustering Validity (Definition 8 equation 3.5) for a given clustering (labels):

        DBCV(C) = \sum_{i=1}^{i=l} \frac{|C_i|}{|O|} V_c(C_i)

        C - Clustering
        l - number of clusters
        |O| - number of elements including noise
        |C_i| - number of elements in cluster i
        V_c (C_i) - Validity index of cluster i

    :param data: dataset array
    :param partition: clustering labels
    :return:
    """
    # cluster labels included in the labels
    clusters = np.unique(partition)

    # checking singleton clusters
    # this function puts all singletons as noise
    for i in range(len(clusters)):
        if np.sum(partition == clusters[i]) == 1:
            partition[partition == clusters[i]] = -1
            clusters[i] = -1
    # all clusters except for -1
    clusters = np.setdiff1d(clusters, -1)
    # if no clusters left or just one cluster left return 0
    if len(clusters) == 0 or len(clusters) == 1:
        return 0

    # exclude noise points from dataset
    data = data[partition != -1, :]
    # calculate squared euclidean distance
    dist = squareform(pdist(data)) ** 2

    # original labelling
    poriginal = partition
    # exclude noise points from labeling
    partition = partition[partition != -1]

    # numbers of clusters
    nclusters = len(clusters)
    # numbers of objects left in data and numbers of features
    nobjects, nfeatures = data.shape

    # initialize all points core distance with zeros
    d_ucore_cl = np.zeros((nobjects))
    # initialize dsc (Density Sparseness of a cluster (Definition 5))
    dsc = np.zeros((nclusters))

    # initialize array for internal edges
    internal_edges = [None] * nclusters
    # initialize array for internal nodes
    int_node_data = [None] * nclusters

    # for each cluster
    for i in range(nclusters):
        # indices of objects in cluster i
        objects_cl = np.where(partition == clusters[i])[0]
        # number of objects in cluster i
        number_of_objects_cl = len(objects_cl)
        # core distances for cluster i
        d_ucore_cl[objects_cl], mr = matrix_mutual_reachability_distance(
            number_of_objects_cl, dist[np.ix_(objects_cl, objects_cl)], nfeatures
        )
        # initialize dict for cluster
        G = {
            "no_vertices": number_of_objects_cl,
            "MST_edges": np.zeros((number_of_objects_cl - 1, 3)),
            "MST_degrees": np.zeros((number_of_objects_cl), dtype=int),
            "MST_parent": np.zeros((number_of_objects_cl), dtype=int),
        }

        #  calculate minimal spanning tree
        Edges, Degrees = MST_Edges(G, 0, mr)
        # extract internal nodes (nodes with at least two adjacent nodes / degree>1)
        internal_nodes = np.where(Degrees != 1)[0]
        # indices of internal edges
        int_edg1 = np.where(np.isin(Edges[:, 0], internal_nodes))[0]
        int_edg2 = np.where(np.isin(Edges[:, 1], internal_nodes))[0]
        # extract internal edges (edge between two internal nodes)
        internal_edges[i] = np.intersect1d(int_edg1, int_edg2)

        # if there are internal edges
        if len(internal_edges[i]) != 0:
            # density sparseness of a cluster:
            # maximum edge weight of the internal edges
            dsc[i] = np.max(Edges[internal_edges[i], 2])
        # if we do not have internal edges
        else:
            # dsc is maximum of all MST edges
            dsc[i] = np.max(Edges[:, 2])
        # data for internal nodes in i
        int_node_data[i] = objects_cl[internal_nodes]
        # if we do not have internal nodes we set it to all objects in cluster
        if len(int_node_data[i]) == 0:
            int_node_data[i] = objects_cl
    # initialize separation point
    sep_point = np.zeros((nobjects, nobjects))
    # for all points in the cluster
    for i in range(nobjects - 1):
        for j in range(i, nobjects):
            #
            sep_point[i, j] = np.max([dist[i, j], d_ucore_cl[i], d_ucore_cl[j]])
            sep_point[j, i] = sep_point[i, j]

    # initialize
    valid = 0
    min_dscp = np.ones((nclusters)) * np.inf
    # for all clusters
    for i in range(nclusters):
        other_cls = np.setdiff1d(clusters, clusters[i])
        # initialize
        dscp = np.zeros(len(other_cls))
        # check all other clusters
        for j in range(len(other_cls)):
            indices = np.where(clusters == other_cls[j])[0]
            # separation between points internal nodes of cluster i and cluster j
            temp = [
                sep_point[e, f]
                for e in int_node_data[i]
                for f in int_node_data[indices[0]]
            ]
            # Density separation of a pair of clusters:
            # minimum reachability distance between the internal nodes (Definition 6)
            dscp[j] = np.min([np.min(temp)])
        # DSCP for cluster i is min over all other clusters
        min_dscp[i] = np.min(dscp)
        # DBCV for this cluster
        dbcvcl = (min_dscp[i] - dsc[i]) / np.max([dsc[i], min_dscp[i]])
        # sum up and factor depending on number of elements in cluster
        valid = valid + (dbcvcl * np.sum([partition == clusters[i]]))
    # divide by number of assigned labels including noise
    valid = valid / len(poriginal)
    return valid
