import numpy


def normalize(trSet, vsSet, tsSet):
    """ Normalization routine for Letter 1, 2 and 3.
    Normalization includes scaling [x, y] coordinates according to the maximum value found in the overall dataset.
    The input sets will be modified in-place.

    Input:
    - tsSet: a dictionary of (training) graphs of the form {id: (graph, label)}
    - vsSet: a dictionary of (validation) graphs of the form {id: (graph, label)}
    - tsSet: a dictionary of (test) graphs of the form {id: (graph, label)}
    Output:
    - normFactor: a normalization factor that will be used in nodeDissimilarity (i.e., square root of the number of components). """

    # Find max value in training set by stacking [x, y] pairs, then flattening and then taking the max
    MAX_trSet = numpy.zeros((1, 2))
    for k in sorted(trSet.keys()):
        # extract graph
        thisGraph = trSet[k][0]
        # append node labels
        for n in thisGraph.nodes():
            MAX_trSet = numpy.vstack((MAX_trSet, thisGraph.nodes[n]['coords']))
    MAX_trSet = MAX_trSet.ravel().max()

    # Find max value in validation set by stacking [x, y] pairs, then flattening and then taking the max
    MAX_vsSet = numpy.zeros((1, 2))
    for k in sorted(vsSet.keys()):
        # extract graph
        thisGraph = vsSet[k][0]
        # append node labels
        for n in thisGraph.nodes():
            MAX_vsSet = numpy.vstack((MAX_vsSet, thisGraph.nodes[n]['coords']))
    MAX_vsSet = MAX_vsSet.ravel().max()

    # Find max value in test set by stacking [x, y] pairs, then flattening and then taking the max
    MAX_tsSet = numpy.zeros((1, 2))
    for k in sorted(tsSet.keys()):
        # extract graph
        thisGraph = tsSet[k][0]
        # append node labels
        for n in thisGraph.nodes():
            MAX_tsSet = numpy.vstack((MAX_tsSet, thisGraph.nodes[n]['coords']))
    MAX_tsSet = MAX_tsSet.ravel().max()

    # Find overall max value
    MAX = max([MAX_trSet, MAX_vsSet, MAX_tsSet])

    # Scale training set
    for k in sorted(trSet.keys()):
        # extract graph
        thisGraph = trSet[k][0]
        # normalise node labels
        for n in thisGraph.nodes():
            thisGraph.nodes[n]['coords'] = thisGraph.nodes[n]['coords'] / MAX
        # push back
        trSet[k] = (thisGraph, trSet[k][1])

    # Scale validation set
    for k in sorted(vsSet.keys()):
        # extract graph
        thisGraph = vsSet[k][0]
        # normalise node labels
        for n in thisGraph.nodes():
            thisGraph.nodes[n]['coords'] = thisGraph.nodes[n]['coords'] / MAX
        # push back
        vsSet[k] = (thisGraph, vsSet[k][1])

    # Scale test set
    for k in sorted(tsSet.keys()):
        # extract graph
        thisGraph = tsSet[k][0]
        # normalise node labels
        for n in thisGraph.nodes():
            thisGraph.nodes[n]['coords'] = thisGraph.nodes[n]['coords'] / MAX
        # push back
        tsSet[k] = (thisGraph, tsSet[k][1])

    normFactor = numpy.sqrt(thisGraph.nodes[n]['coords'].shape[1])
    return normFactor
