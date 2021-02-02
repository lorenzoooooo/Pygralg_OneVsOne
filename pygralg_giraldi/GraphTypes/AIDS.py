# -*- coding: utf-8 -*-
"""
@author: luca
"""

from ast import literal_eval as lit_ev
from numpy import asarray,reshape,linalg


class AIDSdiss:
    
    def __init__(self):
        self._VertexDissWeights=1.0
#        self._EdgeDissWeights=1.0
    
    def nodeDissimilarity(self,a, b):
        alpha=0.1
        beta=0.2
        gamma=0.7 # NON SI SA DOVE LI HANNO TROVATI
        D=0
    
        if(a['charge']!= b['charge']):
            D+=alpha
        if(a['chem']!= b['chem']):
            D+=beta
        
        D+=linalg.norm(a['coords'] - b['coords'])*gamma /self._VertexDissWeights
        return D 
    
    
    def edgeDissimilarity(self,a, b):
        return 0.0


def parser(g):

    for node in g.nodes():
#        list_labels = list(map(lambda x: lit_ev(x), g.nodes[node].values()))
        g.nodes[node]['charge']=int(lit_ev(g.nodes[node]['charge']))
        g.nodes[node]['chem']=int(lit_ev(g.nodes[node]['chem']))
        g.nodes[node]['symbol']=lit_ev(g.nodes[node]['symbol']).strip()
        real_2Dcoords = [float(lit_ev(g.nodes[node]['x'])), float(lit_ev(g.nodes[node]['y']))]        
#        g.nodes[node]['coords'] = reshape(asarray(real_2Dcoords), (1, 2)) ?
        g.nodes[node]['coords'] = reshape(asarray(real_2Dcoords), (2,))
        {g.nodes[node].pop(k) for k in ['y', 'x']}