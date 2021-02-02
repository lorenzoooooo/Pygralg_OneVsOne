## PROVA DELLA REPLICA DELL'ESTRATTORE DI SPARE

import copy

def SpareExtractor(G, desiredOrder):
    # # add single nodes
    # subgraphsList = [[i] for i in G.nodes()]
    # subgraphsList = [G.subgraph(sub_nodes).copy() for sub_nodes in subgraphsList]

    # #
    # for o in range(2, desiredOrder + 1):
    #     subgraphsListNew = []
    #     for i in range(len(subgraphsList)):
    #         thisSubgraph = subgraphsList[i]

    #         # for node in sorted(thisSubgraph.nodes()):
    #         #     adj_nodes_1hop = list(G.neighbors(node))
    #         #     for neighbour_1hop in sorted(adj_nodes_1hop):
    #         #         if neighbour_1hop in thisSubgraph.nodes():
    #         #             pass
    #         #         else:
    #         #             # subgraphsList.append(G.subgraph([node, neighbour_1hop]).copy())
    #         #             thisSubgraph.add_edge(node, neighbour_1hop)
    #         #             subgraphsListNew.append(thisSubgraph)

    #         #             adj_nodes_2hop = list(G.neighbors(neighbour_1hop))
    #         #             for neighbour_2hop in sorted(adj_nodes_2hop):
    #         #                 if neighbour_2hop in thisSubgraph.nodes():
    #         #                     pass
    #         #                 else:
    #         #                     # subgraphsList.append(G.subgraph([neighbour_2hop, neighbour_1hop]).copy())
    #         #                     thisSubgraph.add_edge(neighbour_2hop, neighbour_1hop)
    #         #                     subgraphsListNew.append(thisSubgraph)
    #         for node in sorted(thisSubgraph.nodes()):
    #             adj_nodes_1hop = list(G.neighbors(node))
    #             for neighbour_1hop in sorted(adj_nodes_1hop):
    #                 if neighbour_1hop not in thisSubgraph.nodes():
    #                     thisSubgraph.add_edge(node, neighbour_1hop)
    #                     subgraphsListNew.append(copy.deepcopy(thisSubgraph))

    #                 adj_nodes_2hop = list(G.neighbors(neighbour_1hop))
    #                 for neighbour_2hop in sorted(adj_nodes_2hop):
    #                     if neighbour_2hop not in thisSubgraph.nodes():
    #                         thisSubgraph.add_edge(neighbour_2hop, neighbour_1hop)
    #                         subgraphsListNew.append(copy.deepcopy(thisSubgraph))
    #     subgraphsList = copy.deepcopy(subgraphsListNew)
    #     # subgraphsList = subgraphsList + subgraphsListNew
    # return subgraphsList

    subgraphsList = [[i] for i in G.nodes()]
    for o in range(2, desiredOrder + 1):
        subgraphsListNew = []
        for i in range(len(subgraphsList)):
            thisSubgraph = subgraphsList[i]
            for node in sorted(thisSubgraph):
                adj_nodes_1hop = list(G.neighbors(node))
                for neighbour_1hop in sorted(adj_nodes_1hop):
                    subgraphsListNew.append(copy.deepcopy(thisSubgraph.add_edge(node, neighbour_1hop)))
    