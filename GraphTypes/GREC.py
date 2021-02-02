# -*- coding: utf-8 -*-

from ast import literal_eval as lit_ev
from numpy import asarray,reshape,linalg,abs



class GRECdiss:
    
    def __init__(self):

        self._VertexDissWeights=1.0
        self._EdgeDissWeights=1.0
        self._vParam1=0.9353
        self._eParam1=0.1423
        self._eParam2=0.125
        self._eParam3=0.323
        self._eParam4=0.1069
        
    def nodeDissimilarity(self,a, b):
        D=0
        if(a['type']!=b['type']):
            D+=self._vParam1
        D+=(1-self._vParam1)*linalg.norm(a['coords'] - b['coords']) /  self._VertexDissWeights

        return D
    
    
    def edgeDissimilarity(self,a, b):
        D=0
        if(a['frequency']==b['frequency']):
            if(a['frequency']==1):
                if(a['type0']=="line" and b['type0']=="line"):
                    D=self._eParam1*(abs( a['angle0'] - b['angle0']) +1.575 )/3.15
                elif (a['type0']=="arc" and b['type0']=="arc"):
                    D=self._eParam2*abs(a['angle0']-b['angle0'])/self._EdgeDissWeights
                else:
                    D=self._eParam3
            else:
                if(a['type0']=="line" and b['type0']=="line"):
                    D=self._eParam1*(abs(a['angle0'] - b['angle0']) + 1.575)/(2*3.15) + self._eParam2*abs(a['angle1'] - b['angle1'])/(2*self._EdgeDissWeights)
                elif (a['type0']=="arc" and b['type0']=="arc"):
                    D=self._eParam1*(abs(a['angle1'] - b['angle1']) + 1.575)/(2*3.15) + self._eParam2*abs(a['angle0'] - b['angle0'])/(2*self._EdgeDissWeights)
                else:
                    D=self._eParam3    
        else:
            D=self._eParam4
        
        return D
                    
        
def parser(g):
    # GREC has particular attributes on edges which are not common to all graphs in dataset.
    # In order to manage this issue, when an attribute is not found, it will be added to the graph 
    # with empty strings or 0's depending on its type. 
    # Since NetworkX edges/nodes are dict structure a try/except workaround is employed, then when
    # missing key (KeyError) error occurs, the parser manages the exception adding the key to the edge labels 
    
    # Vertex Attributes:
    # -type(string)
    # -coords (x,y real values)
    # Edge attributes:
    # -frequency (real)
    # -type0,1 (strings)
    # -angle0,1 (real)

    for node in g.nodes():
        g.nodes[node]['type']=g.nodes[node]['type']
        real_2Dcoords = [float(lit_ev(g.nodes[node]['x'])), float(lit_ev(g.nodes[node]['y']))]        
        g.nodes[node]['coords'] = reshape(asarray(real_2Dcoords), (2,))
        {g.nodes[node].pop(k) for k in ['y', 'x']}
        
    for edge in g.edges():
        u=edge[0]
        v=edge[1]
        try:
            g.edges[u,v]['angle0']=float(lit_ev(g.edges[u,v]['angle0']))
        except KeyError:
            g.edges[u,v]['angle0']=0
        try:
            g.edges[u,v]['angle1']=float(lit_ev(g.edges[u,v]['angle1']))
        except KeyError:
            g.edges[u,v]['angle1']=0
        try:
            g.edges[u,v]['type0']=g.edges[u,v]['type0']
        except KeyError:
            g.edges[u,v]['type0']=""
        try:
            g.edges[u,v]['type1']=g.edges[u,v]['type1']
        except KeyError:
            g.edges[u,v]['type1']=""                    
            
        g.edges[u,v]['frequency']=float(lit_ev(g.edges[u,v]['frequency']))
