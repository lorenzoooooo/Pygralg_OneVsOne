# -*- coding: utf-8 -*-
import numpy


def normalize(trSet, vsSet, tsSet):
    """ Normalization routine for AIDS.
    Normalization consists in finding a normalization constant by exploiting node attributes across the entire dataset.
    The input sets remain untouched.

    Input:
    - tsSet: a dictionary of (training) graphs of the form {id: (graph, label)}
    - vsSet: a dictionary of (validation) graphs of the form {id: (graph, label)}
    - tsSet: a dictionary of (test) graphs of the form {id: (graph, label)}
    Output:
    - aidsw: a normalization factor that will be used in nodeDissimilarity. """

    # Init main structure
    Set_stack = numpy.zeros((1, 2))

    # Stack node attributes belonging to training set
    trSet_stack = numpy.zeros((1, 2))
    for k in sorted(trSet.keys()):
        # extract graph
        thisGraph = trSet[k][0]
        # append node labels
        for n in thisGraph.nodes():
            trSet_stack = numpy.vstack((trSet_stack, thisGraph.nodes[n]['coords']))

    # Update main structure
    Set_stack = numpy.vstack((Set_stack, trSet_stack))

    # Stack node attributes belonging to validation set
    vsSet_stack = numpy.zeros((1, 2))
    for k in sorted(vsSet.keys()):
        # extract graph
        thisGraph = vsSet[k][0]
        # append node labels
        for n in thisGraph.nodes():
            vsSet_stack = numpy.vstack((vsSet_stack, thisGraph.nodes[n]['coords']))

    # Update main structure
    Set_stack = numpy.vstack((Set_stack, vsSet_stack))

    # Stack node attributes belonging to test set
    tsSet_stack = numpy.zeros((1, 2))
    for k in sorted(tsSet.keys()):
        # extract graph
        thisGraph = tsSet[k][0]
        # append node labels
        for n in thisGraph.nodes():
            tsSet_stack = numpy.vstack((tsSet_stack, thisGraph.nodes[n]['coords']))

    # Update main structure
    Set_stack = numpy.vstack((Set_stack, tsSet_stack))

    # Find column-wise min and max values
    MIN_SetX, MAX_SetX = Set_stack[:, 0].min(), Set_stack[:, 0].max()
    MIN_SetY, MAX_SetY = Set_stack[:, 1].min(), Set_stack[:, 1].max()

    # Eval normalization factor
    aidsw = numpy.sqrt((MAX_SetX - MIN_SetX)**2 + (MAX_SetY - MIN_SetY)**2)
    return aidsw
