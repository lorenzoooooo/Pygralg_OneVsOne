import itertools

from clusteringAlgorithms import SpareBSAS
from joblib import Parallel, delayed


def ensembleGranulator(bucket, dissimilarityFunction, Q, eta, tau_f, theta_candidates, epsilon, n_jobs=1):
    """ Ensemble BSAS-based granulator.

    Input:
    - bucket: list of graphs to be processed
    - dissimilarityFunction: callable for the dissimilarity measure to be used
    - Q: maximum number of allowed clusters for BSAS
    - eta: tradeoff parameter in cluster quality index F
    - tau_f: threshold for symbol inclusion in alphabet
    - theta_candidates: list of theta values for BSAS ensemble
    - epsilon: tolerance for compactness
    - n_jobs: number of threads for parallel execution (positive integer, defaults to 1 - i.e., no parallelism)
    Output:
    - alphabet: a tuple of the form (symbol, symbol-specific threshold).
    NOTE:
    n_jobs shall be greater than 1 (i.e., parallelisation enabled) only if the granulation has to be performed
    outside other parallel executions (e.g., genetic algorithm), otherwise you'll have nested threads. """

    # Misc
    bucketSize = len(bucket)

    # Run BSAS with different theta values
    # NOTE: Joblib accepts n_jobs=1 and triggers a SequentialBackend, so in principle this if/else can be removed.
    # Yet, in this manner, if n_jobs=1 we do not trigger any Joblib backend, resulting in no overhead at all.
    if n_jobs == 1:
        clusters, representatives, representatives_IDs, clusters_DissimMatrix = [], [], [], []
        for theta_bsas in theta_candidates:
            # run clustering procedure
            this_clusters, this_representatives, this_representatives_IDs, this_clusters_DissimMatrix = SpareBSAS(bucket, theta_bsas, Q, dissimilarityFunction)
            # merge
            clusters = clusters + this_clusters
            representatives = representatives + this_representatives
            representatives_IDs = representatives_IDs + this_representatives_IDs
            clusters_DissimMatrix = clusters_DissimMatrix + this_clusters_DissimMatrix
    else:
        output = Parallel(n_jobs=n_jobs)(delayed(SpareBSAS)(bucket, theta_bsas, Q, dissimilarityFunction) for theta_bsas in theta_candidates)
        # strip the four outputs from SpareBSAS
        clusters = [output[i][0] for i in range(len(theta_candidates))]
        representatives = [output[i][1] for i in range(len(theta_candidates))]
        representatives_IDs = [output[i][2] for i in range(len(theta_candidates))]
        clusters_DissimMatrix = [output[i][3] for i in range(len(theta_candidates))]
        # merge (we don't care about the proper theta value)
        clusters = [item for sublist in clusters for item in sublist]
        representatives = [item for sublist in representatives for item in sublist]
        representatives_IDs = [item for sublist in representatives_IDs for item in sublist]
        clusters_DissimMatrix = [item for sublist in clusters_DissimMatrix for item in sublist]

    # # Discard singleton clusters and universe clusters
    # for i in range(len(clusters)):
    #     if len(clusters[i]) == 1 or len(clusters[i]) == bucketSize:
    #         clusters[i] = None
    #         representatives[i] = None
    #         representatives_IDs[i] = None
    #         clusters_DissimMatrix[i] = None
    # clusters = [item for item in clusters if item is not None]
    # representatives = [item for item in representatives if item is not None]
    # representatives_IDs = [item for item in representatives_IDs if item is not None]
    # clusters_DissimMatrix = [item for item in clusters_DissimMatrix if item is not None]
    #
    # # Evaluate compactness, cardinality and cluster quality index. Eventually promote to alphabet
    # alphabet = []
    # F, compactness, cardinality = [None] * len(clusters), [None] * len(clusters), [None] * len(clusters)
    # for i in range(len(clusters)):
    #     cardinality[i] = 1 - (len(clusters[i]) / bucketSize)
    #     cardinality[i] = cardinality[i] * 0.1   # NOTE: because boh
    #     compactness[i] = sum(clusters_DissimMatrix[i][clusters[i].index(representatives_IDs[i]), :]) / (len(clusters[i]) - 1)
    #     F[i] = eta * compactness[i] + (1 - eta) * cardinality[i]
    #     if F[i] <= tau_f:
    #         alphabet.append((representatives[i], epsilon * compactness[i]))

    alphabet = []
    F, compactness, cardinality = [None] * len(clusters), [None] * len(clusters), [None] * len(clusters)
    for i in range(len(clusters)):
        if len(clusters[i]) == 1 or len(clusters[i]) == bucketSize:
            pass    # discard singleton clusters and universe clusters
        else:
            # evaluate compactness, cardinality and cluster quality index
            cardinality[i] = 1 - (len(clusters[i]) / bucketSize)
            cardinality[i] = cardinality[i] * 0.1   # NOTE: because boh
            compactness[i] = sum(clusters_DissimMatrix[i][clusters[i].index(representatives_IDs[i]), :]) / (len(clusters[i]) - 1)
            F[i] = eta * compactness[i] + (1 - eta) * cardinality[i]
            # eventually promote to alphabet
            if F[i] <= tau_f:
                alphabet.append((representatives[i], epsilon * compactness[i]))

    # Discard duplicates in alphabet
    alphabet_exp = [(item[0].name, item[0], item[1]) for item in alphabet]  # alphabet_exp is a list of tuples of the form (subgraph ID, subgraph, radius)
    alphabet_exp = sorted(alphabet_exp, key=lambda x: x[0])                 # sort according to subgraph IDs so that duplicates are contiguous 
    alphabet_shrink = []
    for k, g in itertools.groupby(alphabet_exp, lambda x: x[0]):
        g = list(g)                         # amongst all subgraphs having the same ID
        g = max(g, key=lambda x: x[-1])     # pick the one with largest radius (i.e., epsilon * compactness)
        alphabet_shrink.append(g[1:])       # and then store, discarding the subgraph ID
    # alphabet_shrink = {}
    # for i in range(len(alphabet)):
    #     alphabet_shrink.setdefault(alphabet[i][0].name, alphabet[i])
    # alphabet_shrink = list(alphabet_shrink.values())
    alphabet = alphabet_shrink

    return alphabet


def ensembleStratifiedGranulator(bucket, dissimilarityFunction, Q, eta, tau_f, numClasses, theta_candidates, epsilon, n_jobs=1):
    """ Ensemble BSAS-based stratified granulator.

    Input:
    - bucket: list-of-lists containing subgraphs of class i in position i
    - dissimilarityFunction: callable for the dissimilarity measure to be used
    - Q: maximum number of allowed clusters for BSAS
    - eta: tradeoff parameter in cluster quality index F
    - tau_f: threshold for symbol inclusion in alphabet
    - numClasses: number of classes for the classification problem
    - n_jobs: number of threads for parallel execution (positive integer, defaults to 1 - i.e., no parallelism)
    Output:
    - alphabet: a tuple of the form (symbol, symbol-specific threshold).
    NOTE:
    n_jobs shall be greater than 1 (i.e., parallelisation enabled) only if the granulation has to be performed
    outside other parallel executions (e.g., genetic algorithm), otherwise you'll have nested threads. """

    # Run BSAS with different theta values for each class-specific bucket
    # NOTE: Joblib accepts n_jobs=1 and triggers a SequentialBackend, so in principle this if/else can be removed.
    # Yet, in this manner, if n_jobs=1 we do not trigger any Joblib backend, resulting in no overhead at all.
    clusters, representatives, representatives_IDs, clusters_DissimMatrix = [], [], [], []
    if n_jobs == 1:
        for c in range(numClasses):
            for theta_bsas in theta_candidates:
                this_clusters, this_representatives, this_representatives_IDs, this_clusters_DissimMatrix = SpareBSAS(bucket[c], theta_bsas, Q, dissimilarityFunction)
                # keep track of partitions and their class label
                this_clusters = list(zip(this_clusters, [c] * len(this_clusters)))
                this_representatives = list(zip(this_representatives, [c] * len(this_representatives)))
                this_representatives_IDs = list(zip(this_representatives_IDs, [c] * len(this_representatives_IDs)))
                this_clusters_DissimMatrix = list(zip(this_clusters_DissimMatrix, [c] * len(this_clusters_DissimMatrix)))
                # merge
                clusters = clusters + this_clusters
                representatives = representatives + this_representatives
                representatives_IDs = representatives_IDs + this_representatives_IDs
                clusters_DissimMatrix = clusters_DissimMatrix + this_clusters_DissimMatrix
    else:
        for c in range(numClasses):
            output = Parallel(n_jobs=n_jobs)(delayed(SpareBSAS)(bucket[c], theta_bsas, Q, dissimilarityFunction) for theta_bsas in theta_candidates)
            # strip the four outputs from SpareBSAS
            this_clusters = [output[i][0] for i in range(len(theta_candidates))]
            this_representatives = [output[i][1] for i in range(len(theta_candidates))]
            this_representatives_IDs = [output[i][2] for i in range(len(theta_candidates))]
            this_clusters_DissimMatrix = [output[i][3] for i in range(len(theta_candidates))]
            # flat (we don't care about the proper theta value)
            this_clusters = [item for sublist in this_clusters for item in sublist]
            this_representatives = [item for sublist in this_representatives for item in sublist]
            this_representatives_IDs = [item for sublist in this_representatives_IDs for item in sublist]
            this_clusters_DissimMatrix = [item for sublist in this_clusters_DissimMatrix for item in sublist]
            # keep track of partitions and their class label
            this_clusters = list(zip(this_clusters, [c] * len(this_clusters)))
            this_representatives = list(zip(this_representatives, [c] * len(this_representatives)))
            this_representatives_IDs = list(zip(this_representatives_IDs, [c] * len(this_representatives_IDs)))
            this_clusters_DissimMatrix = list(zip(this_clusters_DissimMatrix, [c] * len(this_clusters_DissimMatrix)))
            # merge
            clusters = clusters + this_clusters
            representatives = representatives + this_representatives
            representatives_IDs = representatives_IDs + this_representatives_IDs
            clusters_DissimMatrix = clusters_DissimMatrix + this_clusters_DissimMatrix

    # # Discard singleton clusters and universe clusters
    # for i in range(len(clusters)):
    #     if len(clusters[i][0]) == 1 or len(clusters[i][0]) == len(bucket[clusters[i][1]]):
    #         clusters[i] = None
    #         representatives[i] = None
    #         representatives_IDs[i] = None
    #         clusters_DissimMatrix[i] = None
    # clusters = [item for item in clusters if item is not None]
    # representatives = [item for item in representatives if item is not None]
    # representatives_IDs = [item for item in representatives_IDs if item is not None]
    # clusters_DissimMatrix = [item for item in clusters_DissimMatrix if item is not None]
    # 
    # # Evaluate compactness, cardinality and cluster quality index. Eventually promote to alphabet
    # alphabet = []
    # F, compactness, cardinality = [None] * len(clusters), [None] * len(clusters), [None] * len(clusters)
    # for i in range(len(clusters)):
    #     cardinality[i] = 1 - (len(clusters[i][0]) / len(bucket[clusters[i][1]]))
    #     cardinality[i] = cardinality[i] * 0.1   # NOTE: because boh
    #     compactness[i] = sum(clusters_DissimMatrix[i][0][clusters[i][0].index(representatives_IDs[i][0]), :]) / (len(clusters[i][0]) - 1)
    #     F[i] = eta * compactness[i] + (1 - eta) * cardinality[i]
    #     if F[i] <= tau_f:
    #         alphabet.append((representatives[i][0], epsilon * compactness[i]))

    alphabet = []
    F, compactness, cardinality = [None] * len(clusters), [None] * len(clusters), [None] * len(clusters)
    for i in range(len(clusters)):
        if len(clusters[i][0]) == 1 or len(clusters[i][0]) == len(bucket[clusters[i][1]]):
            pass    # discard singleton clusters and universe clusters
        else:
            # evaluate compactness, cardinality and cluster quality index
            cardinality[i] = 1 - (len(clusters[i][0]) / len(bucket[clusters[i][1]]))
            cardinality[i] = cardinality[i] * 0.1   # NOTE: because boh
            compactness[i] = sum(clusters_DissimMatrix[i][0][clusters[i][0].index(representatives_IDs[i][0]), :]) / (len(clusters[i][0]) - 1)
            F[i] = eta * compactness[i] + (1 - eta) * cardinality[i]
            # eventually promote to alphabet
            if F[i] <= tau_f:
                alphabet.append((representatives[i][0], epsilon * compactness[i]))

    # Discard duplicates in alphabet
    alphabet_exp = [(item[0].name, item[0], item[1]) for item in alphabet]  # alphabet_exp is a list of tuples of the form (subgraph ID, subgraph, radius)
    alphabet_exp = sorted(alphabet_exp, key=lambda x: x[0])                 # sort according to subgraph IDs so that duplicates are contiguous 
    alphabet_shrink = []
    for k, g in itertools.groupby(alphabet_exp, lambda x: x[0]):
        g = list(g)                         # amongst all subgraphs having the same ID
        g = max(g, key=lambda x: x[-1])     # pick the one with largest radius (i.e., epsilon * compactness)
        alphabet_shrink.append(g[1:])       # and then store, discarding the subgraph ID
    # alphabet_shrink = {}
    # for i in range(len(alphabet)):
    #     alphabet_shrink.setdefault(alphabet[i][0].name, alphabet[i])
    # alphabet_shrink = list(alphabet_shrink.values())
    alphabet = alphabet_shrink

    return alphabet
