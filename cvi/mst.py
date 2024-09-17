import numpy as np
from scipy.spatial.distance import pdist, squareform

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


