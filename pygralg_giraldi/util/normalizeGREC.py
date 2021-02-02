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

    Set_stackCoords = numpy.zeros((1, 2))
    Set_stackAngles = numpy.zeros((1,))

    sets = [trSet, vsSet, tsSet]

    i = 0
    for thisSet in sets:
        print("Work on ", i)
        for k in sorted(thisSet.keys()):
            # extract graph
            thisGraph = thisSet[k][0]
            # Parsing node
            for n in thisGraph.nodes():
                Set_stackCoords = numpy.vstack((Set_stackCoords, thisGraph.nodes[n]['coords']))
            # Parsing edges
            for e in thisGraph.edges():
                if(thisGraph.edges[e]['frequency'] == 1):
                    if(thisGraph.edges[e]['type0'] == "arc"):
                        Set_stackAngles = numpy.vstack((Set_stackAngles, thisGraph.edges[e]['angle0']))
                else:
                    if(thisGraph.edges[e]['type0'] == "arc"):
                        Set_stackAngles = numpy.vstack((Set_stackAngles, thisGraph.edges[e]['angle0']))
                    else:
                        Set_stackAngles = numpy.vstack((Set_stackAngles, thisGraph.edges[e]['angle1']))
        i += 1

    MINx, MAXx = Set_stackCoords[:, 0].min(), Set_stackCoords[:, 0].max()
    MINy, MAXy = Set_stackCoords[:, 1].min(), Set_stackCoords[:, 1].max()

    MINa, MAXa = Set_stackAngles.min(), Set_stackAngles.max()

    vertexW = numpy.sqrt((MAXx - MINx)**2 + (MAXy - MINy)**2)
    edgeW = numpy.abs(MAXa - MINa)

    return vertexW, edgeW
