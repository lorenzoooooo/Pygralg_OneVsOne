# -*- coding: utf-8 -*-
"""
@author: luca
"""


from ast import literal_eval as lit_ev
from Levenshtein import distance as edit_dist


class PROTEINdiss:

    def __init__(self):

        self._VertexDissWeights = 1.0
        self._EdgeDissWeights = 1.0
        
        self._vParam1 = 1
        
        self._eParam1 = 1
        self._eParam2 = 1
        self._eParam3 = 1
        self._eParam4 = 1
        self._eParam5 = 1
        

    def nodeDissimilarity(self, a, b):
        D = 0
        delta_diss = 0
        if(a['type'] != b['type']):
            delta_diss = 1

        D = self._vParam1 * delta_diss + (1 - self._vParam1) * edit_dist(a['sequence'], b['sequence']) / max(len(a['sequence']), len(b['sequence']))

        return D

    def edgeDissimilarity(self, a, b):
#        D = 0
#        delta_diss_t0 = 0
#        delta_diss_t1 = 0
#        f_a = a['frequency']
#        f_b = b['frequency']
#
#        # Delta diss type labels
#        if(a['type0'] != b['type0']):
#            delta_diss_t0 = 1
#        if(a['type1'] != b['type1']):
#            delta_diss_t1 = 1
#
#        if(f_a == f_b):
#            if(f_a == 1):
#                D = self._eParam1 * delta_diss_t0 + (1 - self._eParam1) * abs(a['distance0'] - b['distance0']) / (self._EdgeDissWeights)
#            else:
#                d0 = self._eParam1 * delta_diss_t0 + (1 - self._eParam1) * abs(a['distance0'] - b['distance0']) / (self._EdgeDissWeights)
#                d1 = self._eParam1 * delta_diss_t1 + (1 - self._eParam1) * abs(a['distance1'] - b['distance1']) / (self._EdgeDissWeights)
#                D = 0.5 * (d0 + d1)
#        else:
#            D = 1
#
#        return D
        
        #5 params
        D = 0
        delta_diss_t0 = 0
        delta_diss_t1 = 0
        f_a = a['frequency']
        f_b = b['frequency']

        # Delta diss type labels
        if(a['type0'] != b['type0']):
            delta_diss_t0 = 1
        if(a['type1'] != b['type1']):
            delta_diss_t1 = 1

        if(f_a == f_b):
            if(f_a == 1):
                D = self._eParam1 * delta_diss_t0 + (1 - self._eParam1) * abs(a['distance0'] - b['distance0']) / (self._EdgeDissWeights)
            else:
                d0 = self._eParam2 * delta_diss_t0 + (1 - self._eParam2) * abs(a['distance0'] - b['distance0']) / (self._EdgeDissWeights)
                d1 = self._eParam3 * delta_diss_t1 + (1 - self._eParam3) * abs(a['distance1'] - b['distance1']) / (self._EdgeDissWeights)
                D = self._eParam4 * (d0 + d1)
        else:
            D = self._eParam5

        return D

def parser(g):
    # Vertex Attributes:
    # -aaLength - (unsigned int)
    # -sequence - (string)
    # -type - (unsigned int)
    # Edge attributes:
    # -distance0 (real)
    # -distance1 (real)
    # -frequency (unsigned inte)
    # -type0 (strings)
    # -type1 (strings)
    for node in g.nodes():
        # temp- they can still be strings
        g.nodes[node]['type'] = int(g.nodes[node]['type'])
        g.nodes[node]['sequence'] = g.nodes[node]['sequence']
        # g.nodes[node]['aaLength']=int(g.nodes[node]['aaLength'])

    for edge in g.edges():
        u = edge[0]
        v = edge[1]
        try:
            g.edges[u, v]['distance0'] = float(lit_ev(g.edges[u, v]['distance0']))
        except KeyError:
            g.edges[u, v]['distance0'] = 0
        try:
            g.edges[u, v]['distance1'] = float(lit_ev(g.edges[u, v]['distance1']))
        except KeyError:
            g.edges[u, v]['distance1'] = 0
        try:
            g.edges[u, v]['type0'] = g.edges[u, v]['type0']
        except KeyError:
            g.edges[u, v]['type0'] = ""
        try:
            g.edges[u, v]['type1'] = g.edges[u, v]['type1']
        except KeyError:
            g.edges[u, v]['type1'] = ""

        g.edges[u, v]['frequency'] = int(g.edges[u, v]['frequency'])