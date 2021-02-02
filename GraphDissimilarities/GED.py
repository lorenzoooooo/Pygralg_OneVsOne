#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 18:13:23 2019

@author: luca
"""

"""
Custom Graph Edit Distance known as node Best Match First.

The function can be initialized with two arguments:
1) A node dissimilarity function
2) An edge dissimilarity function

BMF() return normalized dissimilarity between two graphs.

Implemented as described in:
Bianchi, Filippo Maria, et al. "Granular computing techniques for classification and semantic characterization of structured data." Cognitive Computation 8.3 (2016), pp. 442-461.

Normalisation has been taken from:
Baldini, Luca et al. "Stochastic information granules extraction for graph embedding and classification." In Proceedings of the 11th International Joint Conference on Computational Intelligence, vol. 1 (2019), pp. 391-402.
"""

from copy import deepcopy


class GED:

    def __init__(self, nodeDiss, edgeDiss):

        """ User Defined Node/Edges dissimilarity """
        self.__nodeDiss = nodeDiss
        self.__edgeDiss = edgeDiss

        """Default cost Parameters """
        self.nodesParam = {'sub': 1.0, 'del': 1.0, 'ins': 1.0}
        self.edgesParam = {'sub': 1.0, 'del': 1.0, 'ins': 1.0}

    def BMF(self, g1, g2):

        """ node Best Match First """

        totVertex_DelCost = 0.0
        totVertex_InsCost = 0.0
        totVertex_SubCost = 0.0

        o1 = g1.order()
        o2 = g2.order()

        hash_table = set()  # Best match are evaluated in a single loop
        assignments = {}

        i = 0

        N1 = sorted(g1.nodes())       # store sorted nodes, so we call sorted()
        N2 = sorted(g2.nodes())       # only twice rather than 'o1 + 1' times
        for g1_n in N1:
        
            if(i >= o2):
                break

            minDiss = float("inf")

            for g2_n in N2:

                if g2_n not in hash_table:
                    tmpDiss = self.__nodeDiss(g1.nodes[g1_n], g2.nodes[g2_n])
                    if tmpDiss < minDiss:
                        assigned_id = deepcopy(g2_n)
                        minDiss = tmpDiss
                        assignments[g1_n] = assigned_id

            hash_table.add(assigned_id)

            totVertex_SubCost += minDiss

            i += 1

        if(o1 > o2):
            totVertex_InsCost = abs(o1 - o2)
        else:
            totVertex_DelCost = abs(o2 - o1)

        vertexDiss = self.nodesParam['sub'] * totVertex_SubCost + self.nodesParam['ins'] * totVertex_InsCost + self.nodesParam['del'] * totVertex_DelCost

        """ Edge Induced Matches """

        totEdge_SubCost = 0.0
        totEdge_InsCost = 0.0
        totEdge_DelCost = 0.0
        edgeInsertionCount = 0
        edgeDeletionCount = 0

        edgesIndex1 = 0
        for matchedNodes1 in assignments.items():

            edgesIndex2 = 0
            edge_g1_exist = False
            edge_g2_exist = False

            u_g1 = matchedNodes1[0]
            u_g2 = matchedNodes1[1]

            for matchedNodes2 in assignments.items():

                if matchedNodes1 != matchedNodes2 and edgesIndex2 <= edgesIndex1:

                    v_g1 = matchedNodes2[0]
                    v_g2 = matchedNodes2[1]

                    edge_g1_exist = g1.has_edge(u_g1, v_g1)
                    edge_g2_exist = g2.has_edge(u_g2, v_g2)

                    if edge_g1_exist and edge_g2_exist:
                        # totEdge_SubCost += self.__edgeDiss((u_g1, v_g1), (u_g2, v_g2))
                        totEdge_SubCost += self.__edgeDiss(g1.edges[(u_g1, v_g1)], g2.edges[(u_g2, v_g2)])                        
                    elif edge_g1_exist:
                        edgeInsertionCount += 1
                    elif edge_g2_exist:
                        edgeDeletionCount += 1

                edgesIndex2 += 1

            edgesIndex1 += 1

        edgeDiss = self.edgesParam['sub'] * totEdge_SubCost + self.edgesParam['ins'] * edgeInsertionCount + self.edgesParam['del'] * edgeDeletionCount

        # try:
        #     u = max(o1, o2) + (min(o1, o2) * (min(o1, o2) - 1) / 2)
        # except:
        #     u = 1
        # return (vertexDiss + edgeDiss) / u

        """ Normalise stuff """
        normaliseFactor_vertex = max(o1, o2)
        normaliseFactor_edge = 0.5 * (min(o1, o2) * (min(o1, o2) - 1))

        vertexDiss_norm = vertexDiss / normaliseFactor_vertex
        edgeDiss_norm = edgeDiss if normaliseFactor_edge == 0 else edgeDiss / normaliseFactor_edge

        return 0.5 * (vertexDiss_norm + edgeDiss_norm)
