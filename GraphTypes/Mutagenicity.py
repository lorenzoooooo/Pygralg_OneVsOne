# -*- coding: utf-8 -*-
"""
@author: luca
"""
# from ast import literal_eval as lit_ev
# from numpy import asarray,reshape,linalg


class MUTdiss:

    def __init__(self):
        self._VertexDissWeights = 1.0
        self._EdgeDissWeights = 1.0

    def nodeDissimilarity(self, a, b):
        D = 0
        if(a['chem'] != b['chem']):
            D = 1
        return D

    def edgeDissimilarity(self, a, b):
        D = 0
        if(a['valence'] != b['valence']):
            D = 1
        return D


def parser(g):

    # Chem are string no need parsing
    for edge in g.edges():
        u = edge[0]
        v = edge[1]
        g.edges[u, v]['valence'] = int(g.edges[u, v]['valence'])
