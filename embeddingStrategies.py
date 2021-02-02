import numpy
from joblib import Parallel, delayed


def symbolicHistogramsEmbedder(G_set, alphabet, tau_k, dissimilarityFunction, n_jobs=1):
    """ Embedding via symbolic histograms.

    Input:
    - G_set: a dictionary of (expanded) graphs of the form {id: (list-of-subgraphs, label)}
    - alphabet: list of symbols
    - tau_k: list of symbol-specific thresholds for scoring a match
    - dissimilarityFunction: callable for the subgraphs-vs-alphabet dissimilarity measure to be used
    - n_jobs: number of threads for parallel execution (positive integer, defaults to 1 - i.e., no parallelism)
    Output:
    - tuple containing the instance matrix and label vector.
    NOTE:
    n_jobs shall be greater than 1 (i.e., parallelisation enabled) only if the embedding has to be performed
    outside other parallel executions (e.g., genetic algorithm), otherwise you'll have nested threads. """

    # Misc
    G_set_size = len(G_set)
    M = len(alphabet)

    # NOTE: Joblib accepts n_jobs=1 and triggers a SequentialBackend, so in principle this if/else can be removed.
    # Yet, in this manner, if n_jobs=1 we do not trigger any Joblib backend, resulting in no overhead at all.
    if n_jobs == 1:
        # preallocate
        InstanceMatrix = numpy.zeros((G_set_size, M))
        LabelVector = numpy.zeros(G_set_size)
        thisGraph_embedded = numpy.zeros(M)
        # for each graph
        for k in sorted(G_set.keys()):
            # grab current graph
            thisGraph_expanded = G_set[k][0]
            thisLabel = G_set[k][1]
            # symbolic histogram
            for i in range(M):
                occurrences = [dissimilarityFunction(subgraph, alphabet[i]) <= tau_k[i] for subgraph in thisGraph_expanded]
                thisGraph_embedded[i] = sum(occurrences)
            # store
            InstanceMatrix[k, :] = thisGraph_embedded     # instance matrix
            LabelVector[k] = thisLabel                    # label vector
    else:
        # declare a dummy function that works on a single graph
        def helper(k):
            # grab current graph
            thisGraph_expanded = G_set[k][0]
            thisLabel = G_set[k][1]
            # symbolic histogram
            thisGraph_embedded = [None] * M
            for i in range(M):
                occurrences = [dissimilarityFunction(subgraph, alphabet[i]) <= tau_k[i] for subgraph in thisGraph_expanded]
                thisGraph_embedded[i] = sum(occurrences)
            # return the ID of the graph, its vectorial representation, its label
            return [k] + thisGraph_embedded + [thisLabel]
        # trigger Joblib
        output = Parallel(n_jobs=n_jobs)(delayed(helper)(k) for k in sorted(G_set.keys()))
        # process output
        output = numpy.array(output)
        # output = output[output[:, 0].argsort()]                                   # sort according to graph ID (not needed: Parallel already returns output ordered according to input)
        LabelVector = output[:, -1]                                                 # last column is label
        InstanceMatrix = numpy.delete(output, [0, output.shape[1] - 1], axis=1)     # remove graph ID (0-th column) and label (last column)

    return InstanceMatrix, LabelVector
