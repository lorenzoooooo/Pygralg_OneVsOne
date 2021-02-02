# -*- coding: utf-8 -*-
import numpy

def normalize(trSet, vsSet, tsSet):
    """ Normalization routine for GREC.
    Normalization consists in finding normalization constants by exploiting node/edge attributes across the entire dataset.
    The input sets remain untouched.

    Input:
    - tsSet: a dictionary of (training) graphs of the form {id: (graph, label)}
    - vsSet: a dictionary of (validation) graphs of the form {id: (graph, label)}
    - tsSet: a dictionary of (test) graphs of the form {id: (graph, label)}
    Output:
    - vertexW: a normalization factor that will be used in nodeDissimilarity
    - edgeW: a normalization factor that will be used in edgeDissimilarity. """

    Set_stackDistances = numpy.zeros((1,))

    sets = [trSet, vsSet, tsSet]

    i = 0
    for thisSet in sets:
        print("Work on ", i)
        for k in sorted(thisSet.keys()):
            # extract graph
            thisGraph = thisSet[k][0]
            # Parsing edges
            for e in thisGraph.edges():
                Set_stackDistances = numpy.vstack((Set_stackDistances, thisGraph.edges[e]['distance0']))
                Set_stackDistances = numpy.vstack((Set_stackDistances, thisGraph.edges[e]['distance1']))
        i += 1

    protw = Set_stackDistances.max()

    return protw
