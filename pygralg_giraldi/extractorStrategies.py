import networkx
import itertools
import random


# def exhaustiveExtractor(G, maxOrder, isConnected):
#     """ Combinatorial extractor for undirected graphs.
#
#     Input:
#     - G: a graph
#     - maxOrder: the maximum number of nodes
#     - isConnected: boolean flag that returns connected subgraphs only (if True)
#     Output:
#     - list of subgraphs up to maxOrder nodes.
#     NOTE: This is painfully slow for large graphs/large datasets, use the path-based undirected graphs extractor instead. """
#
#     subgraphs = list()
#     for o in range(1, maxOrder + 1):                            # for each order up to the maximum 'maxOrder'
#         for sub_nodes in itertools.combinations(G.nodes(), o):  # grab combination of nodes for a given order
#             thisSubgraph = G.subgraph(sub_nodes).copy()         # extract subgraph + inherit nodes/edges labels
#             # if the user sets 'isConnected' as 'False', we add everything
#             if isConnected is False:
#                 subgraphs.append(thisSubgraph)
#             # if the user sets 'isConnected' as 'True', we add only if the subgraph is indeed connected
#             elif isConnected is True and networkx.is_connected(thisSubgraph) is True:
#                 subgraphs.append(thisSubgraph)
#
#     return subgraphs


def exhaustiveExtractor(G, maxOrder, isConnected=True):
    """ Path-based extractor for undirected graphs.

    Input:
    - G: a graph
    - maxOrder: the maximum number of nodes
    - isConnected is ignored (it's just for consistency with combinatorial extractor).
    Output:
    - list of subgraphs up to maxOrder nodes. """

    # add nodes
    paths = [[i] for i in G.nodes()]
    # add paths
    for i, j in itertools.product(G.nodes(), G.nodes()):
        paths = paths + list(networkx.all_simple_paths(G, source=i, target=j, cutoff=maxOrder - 1))
    # remove duplicates
    paths = [sorted(item) for item in paths]
    paths = [tuple(item) for item in paths]
    paths = list(set(paths))
    paths = [list(item) for item in paths]
    # inherit nodes/edges labels (and graph name)
    subgraphs = [G.subgraph(sub_nodes).copy() for sub_nodes in paths]
    # assign unique IDs to subgraphs by concatenating '_subgX' to graph name, where X is a sequential integer
    for i in range(len(subgraphs)):
        subgraphs[i].name = subgraphs[i].name + '_subg' + str(i)
    return subgraphs


def exhaustiveExtractor_DiGraphs(G, maxOrder):
    """ Path-based extractor for directed graphs.

    Input:
    - G: a graph
    - maxOrder: the maximum number of nodes
    Output:
    - list of subgraphs up to maxOrder nodes.
    NOTE:
    exhaustiveExtractor() might not return all subgraphs is G is directed.
    This version follows BFS paths, hence considers directed edges. """

    # add nodes
    paths = [[i] for i in G.nodes()]
    # add paths
    for i, j in itertools.product(G.nodes(), G.nodes()):
        paths = paths + list(networkx.all_simple_paths(G, source=i, target=j, cutoff=maxOrder - 1))
    # inherit nodes/edges labels (and graph name)
    subgraphs = [G.subgraph(sub_nodes).copy() for sub_nodes in paths]
    # assign unique IDs to subgraphs by concatenating '_subgX' to graph name, where X is a sequential integer
    for i in range(len(subgraphs)):
        subgraphs[i].name = subgraphs[i].name + '_subg' + str(i)
    return subgraphs


def cliqueExtractor(G, maximal):
    """ Clique-based extractor for undirected graphs.

    Input:
    - G: a graph
    - maximal: a boolean flag that returns only maximal cliques (if True)
    Output:
    - list of its cliques. """

    # extract cliques
    if maximal is True:
        cliques = list(networkx.find_cliques(G))
    elif maximal is False:
        cliques = list(networkx.enumerate_all_cliques(G))
    # inherit nodes/edges labels (and graph name)
    cliques = [G.subgraph(clique).copy() for clique in cliques]
    # assign unique IDs to subgraphs by concatenating '_subgX' to graph name, where X is a sequential integer
    for i in range(len(cliques)):
        cliques[i].name = cliques[i].name + '_subg' + str(i)
    return cliques


def dfsExtractor(G, maxOrder):
    """ Depth-First Search extractor.

    Input:
    - G: a graph
    - maxOrder: the maximum number of nodes
    Output:
    - list of its subgraphs extracted via DFS traverse.
    NOTE: this procedure is recommended for embedder only, as discussed in IJCCI 2019. """

    paths = []

    # start scanning nodes
    for n in G.nodes():
        # n is the candidate seed node: let's check if present in one of the already-found paths
        isPresent = False
        for i in range(len(paths)):
            if n in paths[i]:
                isPresent = True
                break
        # if found, then it will not be a seed node
        if isPresent == True:
            continue
        # otherwise, it will be a seed node
        thisPath = list(networkx.dfs_edges(G, source=n))    # list of edges of the form
        thisPath = thisPath[0:maxOrder - 1]                 # (n0, n1), (n1, n2), ...
        if thisPath == []:
            continue
        thisPath = [list(thisPath[0])] + [[thisPath[i][1]] for i in range(1, len(thisPath))]    # convert to node sequence
        thisPath = [item for sublist in thisPath for item in sublist]                           # n0, n1, n2, ...
        paths = paths + [thisPath]
    
    # inherit nodes/edges labels (and graph name)
    subgraphs = [G.subgraph(sub_nodes).copy() for sub_nodes in paths]
    # assign unique IDs to subgraphs by concatenating '_subgX' to graph name, where X is a sequential integer
    for i in range(len(subgraphs)):
        subgraphs[i].name = subgraphs[i].name + '_subg' + str(i)
    return subgraphs


def bfsExtractor(G, maxOrder):
    """ Breadth-First Search extractor.

    Input:
    - G: a graph
    - maxOrder: the maximum number of nodes
    Output:
    - list of its subgraphs extracted via BFS traverse.
    NOTE: this procedure is recommended for embedder only, as discussed in IJCCI 2019. """

    paths = []

    # start scanning nodes
    for n in G.nodes():
        # n is the candidate seed node: let's check if present in one of the already-found paths
        isPresent = False
        for i in range(len(paths)):
            if n in paths[i]:
                isPresent = True
                break
        # if found, then it will not be a seed node
        if isPresent == True:
            continue
        # otherwise, it will be a seed node
        thisPath = list(networkx.bfs_edges(G, source=n))    # list of edges of the form
        thisPath = thisPath[0:maxOrder - 1]                 # (n0, n1), (n1, n2), ...
        if thisPath == []:
            continue
        thisPath = [list(thisPath[0])] + [[thisPath[i][1]] for i in range(1, len(thisPath))]    # convert to node sequence
        thisPath = [item for sublist in thisPath for item in sublist]                           # n0, n1, n2, ...
        paths = paths + [thisPath]
    
    # inherit nodes/edges labels (and graph name)
    subgraphs = [G.subgraph(sub_nodes).copy() for sub_nodes in paths]
    # assign unique IDs to subgraphs by concatenating '_subgX' to graph name, where X is a sequential integer
    for i in range(len(subgraphs)):
        subgraphs[i].name = subgraphs[i].name + '_subg' + str(i)
    return subgraphs