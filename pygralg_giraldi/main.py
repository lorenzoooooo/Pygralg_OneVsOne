#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:18:05 2019

@author: luca
"""
#-dataName Letter1 -runID 1 -extractG stratPaths -extractE paths -W 1000
from GraphLoaders.DotLoader import DotLoader
from GraphDissimilarities.GED import GED
from util.misc import clipper, BSP

# from geneticAlgorithms import fitnessfunction_GA1, fitnessfunction_GA2, setup_GA1, setup_GA2
from geneticAlgorithms import fitnessfunction_GA1_DE, fitnessfunction_GA2_DE, setup_GA1_DE, setup_GA2_DE
from extractorStrategies import exhaustiveExtractor, cliqueExtractor, bfsExtractor, dfsExtractor
from granulationStrategies import ensembleGranulator, ensembleStratifiedGranulator
from embeddingStrategies import symbolicHistogramsEmbedder

from itertools import combinations

import Graph_Sampling

import sys
import networkx
import pickle
import collections
import random
import time
import numpy
import copy
import os

#from deap import algorithms

from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from scipy.optimize import differential_evolution
from joblib import Parallel, delayed


""" Strip input from command line """
# example usage: python3 main.py -dataName AIDS -runID 1 -extractG stratCliques -extractE cliques -W 10000

arguments = sys.argv

###For DEBUG
# dataName = 'GREC'
# runID = 1
# extractStrategy_Granulator = 'stratSamplePaths'
# extractStrategy_Embedder = 'bfs'
# W = 2100

### Disable command line for DEBUG
dataName = arguments[arguments.index('-dataName') + 1]                      # dataset name
runID = arguments[arguments.index('-runID') + 1]                            # ID of the simulation (this only impacts the name of the pickle file containing the results)
extractStrategy_Granulator = arguments[arguments.index('-extractG') + 1]    # options are: 'paths', 'samplePaths', 'stratPaths', 'stratSamplePaths', 'stratCliques', 'stratSampleCliques'
extractStrategy_Embedder = arguments[arguments.index('-extractE') + 1]      # options are: 'paths', 'cliques', 'dfs', 'bfs'
W = int(arguments[arguments.index('-W') + 1])                               # total number of subgraphs for granulation (only useful for 'stratSampleCliques', 'samplePaths', 'stratSamplePaths')


""" Set some useful parameters """                              # NOTE: some of them can be moved to command line
sampleStrategy = Graph_Sampling.SRW_RWF_ISRW()                  # declare the sampling strategy (ignored for exhaustive- and clique-based strategies)
subgraphsOrder = 5                                              # max subgraphs order (only if cliques are NOT involved)
theta_candidates = BSP(0, 1, 0.1)                               # list of theta candidates for BSAS
epsilon = 1.1                                                   # tolerance value in symbols recognition
#n_threads = 56                                                  # number of threads for parallel execution
n_threads = 1
delimiters = "_", "."                                           # Name file contains id and label


""" Set metric and problem """
if dataName == 'AIDS':
    from util.normalizeAIDS import normalize
    from GraphTypes import AIDS

    graphDissimilarity = AIDS.AIDSdiss()
    parserFunction = AIDS.parser
elif dataName in ['Letter1', 'Letter2', 'Letter3']:
    from util.normalizeLetter import normalize
    from GraphTypes import Letter

    graphDissimilarity = Letter.LETTERdiss()
    parserFunction = Letter.parser
elif dataName == 'GREC':
    from util.normalizeGREC import normalize
    from GraphTypes import GREC

    graphDissimilarity = GREC.GRECdiss()
    parserFunction = GREC.parser
elif dataName == 'PROTEIN':
    from util.normalizePROTEIN import normalize
    from GraphTypes import PROTEIN
    graphDissimilarity = PROTEIN.PROTEINdiss()
    parserFunction = PROTEIN.parser
elif dataName == 'Mutagenicity':
    from GraphTypes import Mutagenicity
    graphDissimilarity = Mutagenicity.MUTdiss()
    parserFunction = Mutagenicity.parser
else:
    raise ValueError('Unknown dataset name.')


""" Define folder """
if dataName == 'AIDS':
#    trDir = "/home/LabRizzi/Documents/Alessio_Martino/dataset_IAM/AIDS/Training/"                                                    # paths DIETrack1, 2 and 3
#    vsDir = "/home/LabRizzi/Documents/Alessio_Martino/dataset_IAM/AIDS/Validation/"
#    tsDir = "/home/LabRizzi/Documents/Alessio_Martino/dataset_IAM/AIDS/Test/"
    trDir = "C:/Users/Utente/Documents/Lorenzo/materiale_didattico_sapienza/lezioni/PR/pyGRALG_LML-master/lorenzo/dataset_IAM/AIDS/Training/"                                                    # paths DIETrack1, 2 and 3
    vsDir = "C:/Users/Utente/Documents/Lorenzo/materiale_didattico_sapienza/lezioni/PR/pyGRALG_LML-master/lorenzo/dataset_IAM/AIDS/Validation/"
    tsDir = "C:/Users/Utente/Documents/Lorenzo/materiale_didattico_sapienza/lezioni/PR/pyGRALG_LML-master/lorenzo/dataset_IAM/AIDS/Test/"
elif dataName == 'Letter1':
#     trDir = "/home/LabRizzi/Documents/Alessio_Martino/dataset_IAM/Letter1/Training/"                                                 # paths DIETrack1, 2 and 3
#     vsDir = "/home/LabRizzi/Documents/Alessio_Martino/dataset_IAM/Letter1/Validation/"
#     tsDir = "/home/LabRizzi/Documents/Alessio_Martino/dataset_IAM/Letter1/Test/"
    trDir = "C:/Users/Utente/Documents/Lorenzo/materiale_didattico_sapienza/lezioni/PR/pyGRALG_LML-master/dataset_IAM/Letter1/Training/"                                                    # paths DIETrack1, 2 and 3
    vsDir = "C:/Users/Utente/Documents/Lorenzo/materiale_didattico_sapienza/lezioni/PR/pyGRALG_LML-master/dataset_IAM/Letter1/Validation/"
    tsDir = "C:/Users/Utente/Documents/Lorenzo/materiale_didattico_sapienza/lezioni/PR/pyGRALG_LML-master/dataset_IAM/Letter1/Test/"
    # trDir = "/home/LabRizzi/Documents/Luca_Baldini/IAM/IAMds/Letter1/Training/"                                                 # paths DIETrack1, 2 and 3
    # vsDir = "/home/LabRizzi/Documents/Luca_Baldini/IAM/IAMds/Letter1/Validation/"
    # tsDir = "/home/LabRizzi/Documents/Luca_Baldini/IAM/IAMds/Letter1/Test/"
elif dataName == 'Letter2':
#     trDir = "/home/LabRizzi/Documents/Alessio_Martino/dataset_IAM/Letter2/Training/"                                                 # paths DIETrack1, 2 and 3
#     vsDir = "/home/LabRizzi/Documents/Alessio_Martino/dataset_IAM/Letter2/Validation/"
#     tsDir = "/home/LabRizzi/Documents/Alessio_Martino/dataset_IAM/Letter2/Test/"
    trDir = "/home/LabRizzi/Documents/Luca_Baldini/IAM/IAMds/Letter2/Training/"                                                 # paths DIETrack1, 2 and 3
    vsDir = "/home/LabRizzi/Documents/Luca_Baldini/IAM/IAMds/Letter2/Validation/"
    tsDir = "/home/LabRizzi/Documents/Luca_Baldini/IAM/IAMds/Letter2/Test/"
elif dataName == 'Letter3':
#     trDir = "/home/LabRizzi/Documents/Alessio_Martino/dataset_IAM/Letter3/Training/"                                                 # paths DIETrack1, 2 and 3
#     vsDir = "/home/LabRizzi/Documents/Alessio_Martino/dataset_IAM/Letter3/Validation/"
#     tsDir = "/home/LabRizzi/Documents/Alessio_Martino/dataset_IAM/Letter3/Test/"
    trDir = "/home/LabRizzi/Documents/Luca_Baldini/IAM/IAMds/Letter3/Training/"                                                 # paths DIETrack1, 2 and 3
    vsDir = "/home/LabRizzi/Documents/Luca_Baldini/IAM/IAMds/Letter3/Validation/"
    tsDir = "/home/LabRizzi/Documents/Luca_Baldini/IAM/IAMds/Letter3/Test/"
elif dataName == 'GREC':
#     trDir = "/home/LabRizzi/Documents/Alessio_Martino/dataset_IAM/GREC/Training/"                                                    # paths DIETrack1, 2 and 3
#     vsDir = "/home/LabRizzi/Documents/Alessio_Martino/dataset_IAM/GREC/Validation/"
#     tsDir = "/home/LabRizzi/Documents/Alessio_Martino/dataset_IAM/GREC/Test/"
    trDir = "/home/LabRizzi/Documents/Luca_Baldini/IAM/IAMds/GREC/Training/"                                                 # paths DIETrack1, 2 and 3
    vsDir = "/home/LabRizzi/Documents/Luca_Baldini/IAM/IAMds/GREC/Validation/"
    tsDir = "/home/LabRizzi/Documents/Luca_Baldini/IAM/IAMds/GREC/Test/"
    # trDir = "/home/luca/Documenti/GRALG_dataset/IAM/IAMds/GREC/Training/"                                              # paths Luca
    # vsDir = "/home/luca/Documenti/GRALG_dataset/IAM/IAMds/GREC/Validation/"
    # tsDir = "/home/luca/Documenti/GRALG_dataset/IAM/IAMds/GREC/Test/"
elif dataName == 'PROTEIN':
#     trDir = "/home/LabRizzi/Documents/Alessio_Martino/dataset_IAM/Protein/Training/"                                                    # paths DIETrack1, 2 and 3
#     vsDir = "/home/LabRizzi/Documents/Alessio_Martino/dataset_IAM/Protein/Validation/"
#     tsDir = "/home/LabRizzi/Documents/Alessio_Martino/dataset_IAM/Protein/Test/"
    trDir = "C:/Users/Utente/Documents/Lorenzo/materiale_didattico_sapienza/lezioni/PR/pyGRALG_LML-master/lorenzo/dataset_IAM/Protein/Training/"                                                    # paths DIETrack1, 2 and 3
    vsDir = "C:/Users/Utente/Documents/Lorenzo/materiale_didattico_sapienza/lezioni/PR/pyGRALG_LML-master/lorenzo/dataset_IAM/Protein/Validation/"
    tsDir = "C:/Users/Utente/Documents/Lorenzo/materiale_didattico_sapienza/lezioni/PR/pyGRALG_LML-master/lorenzo/dataset_IAM/Protein/Test/"
    # trDir = "/home/luca/Documenti/GRALG_dataset/IAM/IAMds/Protein/Training/"                                              # paths Luca
    # vsDir = "/home/luca/Documenti/GRALG_dataset/IAM/IAMds/Protein/Validation/"
    # tsDir = "/home/luca/Documenti/GRALG_dataset/IAM/IAMds/Protein/Test/"
elif dataName == 'Mutagenicity':
#     trDir = "/home/LabRizzi/Documents/Alessio_Martino/dataset_IAM/Mutagenicity/Training/"                                                    # paths DIETrack1, 2 and 3
#     vsDir = "/home/LabRizzi/Documents/Alessio_Martino/dataset_IAM/Mutagenicity/Validation/"
#     tsDir = "/home/LabRizzi/Documents/Alessio_Martino/dataset_IAM/Mutagenicity/Test/"
    trDir = "C:/Users/Utente/Documents/Lorenzo/materiale_didattico_sapienza/lezioni/PR/pyGRALG_LML-master/lorenzo/dataset_IAM/Mutagenicity/Training/"                                                    # paths DIETrack1, 2 and 3
    vsDir = "C:/Users/Utente/Documents/Lorenzo/materiale_didattico_sapienza/lezioni/PR/pyGRALG_LML-master/lorenzo/dataset_IAM/Mutagenicity/Validation/"
    tsDir = "C:/Users/Utente/Documents/Lorenzo/materiale_didattico_sapienza/lezioni/PR/pyGRALG_LML-master/lorenzo/dataset_IAM/Mutagenicity/Test/"
    # trDir = "/home/luca/Documenti/GRALG_dataset/IAM/IAMds/Mutagenicity/Training/"                                              # paths Luca
    # vsDir = "/home/luca/Documenti/GRALG_dataset/IAM/IAMds/Mutagenicity/Validation/"
    # tsDir = "/home/luca/Documenti/GRALG_dataset/IAM/IAMds/Mutagenicity/Test/"


fold = [trDir, vsDir, tsDir]


""" Load Dataset """
t = time.time()
GraphLoader = DotLoader(parserFunction, delimiters)

output = Parallel(n_jobs=3)(delayed(GraphLoader.load)(folder) for folder in fold)                    # here we use 3 (fixed) threads because we have 3 folders
# for folder in fold:
#     if folder == trDir:
#         trSet = GraphLoader.load(folder)
#     elif folder == vsDir:
#         vsSet = GraphLoader.load(folder)
#     elif folder == tsDir:
#         tsSet = GraphLoader.load(folder)
trSet, vsSet, tsSet = output

trSet_size = len(trSet)
vsSet_size = len(vsSet)
tsSet_size = len(tsSet)


""" Normalize """
if dataName == 'AIDS':
    normFactorVertex = normalize(trSet, vsSet, tsSet)
    graphDissimilarity._VertexDissWeights = normFactorVertex
elif dataName in ['Letter1', 'Letter2', 'Letter3']:
    normFactor = normalize(trSet, vsSet, tsSet)
    graphDissimilarity._VertexDissWeights = normFactor
elif dataName == 'GREC':
    normFactorVertex, normFactorEdge = normalize(trSet, vsSet, tsSet)
    graphDissimilarity._VertexDissWeights = normFactorVertex
    graphDissimilarity._EdgeDissWeights = normFactorEdge
elif dataName == 'PROTEIN':
    normFactor = normalize(trSet, vsSet, tsSet)
    graphDissimilarity._EdgeDissWeights = normFactor

""" Minor Processing """
uniqueClasses = list(trSet.values()) + list(vsSet.values()) + list(tsSet.values())                  # convert class labels
uniqueClasses = sorted(list(set([item[1] for item in uniqueClasses])))                              # to sequential integers
numClasses = len(uniqueClasses)                                                                     #
classMapper = {uniqueClasses[i]: i for i in range(numClasses)}                                      #
for k in sorted(trSet.keys()):                                                                      #
    trSet[k] = (trSet[k][0], classMapper[trSet[k][1]])                                              #
for k in sorted(vsSet.keys()):                                                                      #
    vsSet[k] = (vsSet[k][0], classMapper[vsSet[k][1]])                                              #
for k in sorted(tsSet.keys()):                                                                      #
    tsSet[k] = (tsSet[k][0], classMapper[tsSet[k][1]])                                              #

i = 0                                                                                               # make sure that
for k in sorted(trSet.keys()):                                                                      # dictionary keys
    trSet[i] = trSet.pop(k)                                                                         # are sequential
    i = i + 1                                                                                       #
i = 0                                                                                               #
for k in sorted(vsSet.keys()):                                                                      #
    vsSet[i] = vsSet.pop(k)                                                                         #
    i = i + 1                                                                                       #
i = 0                                                                                               #
for k in sorted(tsSet.keys()):                                                                      #
    tsSet[i] = tsSet.pop(k)                                                                         #
    i = i + 1                                                                                       #

for k in sorted(trSet.keys()):                                                                      # assign unique names
    thisGraph = trSet[k][0]                                                                         # to graphs
    thisGraph.name = 'Tr' + str(k)                                                                  # in all of the three sets
    trSet[k] = (thisGraph, trSet[k][1])                                                             #
for k in sorted(vsSet.keys()):                                                                      # each name has the form
    thisGraph = vsSet[k][0]                                                                         # 'TrX', 'VlX', 'TsX'
    thisGraph.name = 'Vl' + str(k)                                                                  # depending on whether the graph belongs to
    vsSet[k] = (thisGraph, vsSet[k][1])                                                             # training, validation or test set,
for k in sorted(tsSet.keys()):                                                                      # with X being a unique numeric ID within
    thisGraph = tsSet[k][0]                                                                         # each set.
    thisGraph.name = 'Ts' + str(k)                                                                  #
    tsSet[k] = (thisGraph, tsSet[k][1])                                                             #

if extractStrategy_Granulator in ['stratSamplePaths', 'stratSampleCliques']:
    classFrequency_trSet = dict(collections.Counter([item[1] for item in list(trSet.values())]))    # let us evaluate
    classFrequency_vsSet = dict(collections.Counter([item[1] for item in list(vsSet.values())]))    # the frequency
    classFrequency_tsSet = dict(collections.Counter([item[1] for item in list(tsSet.values())]))    # of each class
    for k in classFrequency_trSet.keys():                                                           # within each set
        classFrequency_trSet[k] = classFrequency_trSet[k] / trSet_size                              #
    for k in classFrequency_vsSet.keys():                                                           # this is useful
        classFrequency_vsSet[k] = classFrequency_vsSet[k] / vsSet_size                              # for stratified sampling
    for k in classFrequency_tsSet.keys():                                                           # strategies
        classFrequency_tsSet[k] = classFrequency_tsSet[k] / tsSet_size                              #

elapsedTime_Loader = time.time() - t
print("Loaded " + str(trSet_size) + " graphs for Training Set.")
print("Loaded " + str(vsSet_size) + " graphs for Validation Set.")
print("Loaded " + str(tsSet_size) + " graphs for Test Set.")
print("Elapsed Time [Loader]: " + str(elapsedTime_Loader) + " seconds.")
print("\n")


""" Extract for Granulator """
t = time.time()
if extractStrategy_Granulator == 'paths':
    # bucket = [None] * len(trSet)
    # for k in sorted(trSet.keys()):                          # for each training graph
    #     if k % 100 == 0:
    #         print("Working on graph " + str(k + 1) + " out of " + str(trSet_size))
    #     bucket[k] = exhaustiveExtractor(trSet[k][0], subgraphsOrder, isConnected=True)
    bucket = Parallel(verbose=10, n_jobs=n_threads)(delayed(exhaustiveExtractor)(trSet[k][0], subgraphsOrder, isConnected=True) for k in sorted(trSet.keys()))
    bucket = [item for sublist in bucket for item in sublist]
elif extractStrategy_Granulator == 'stratPaths':
    bucket = [None] * numClasses     #dichiaro array di numClasses elementi
    for thisClass in range(numClasses):
        print("Working on class " + str(thisClass + 1) + " out of " + str(numClasses))
        # strip training set patterns belonging to current class
        thisSubset = [item[0] for item in list(trSet.values()) if item[1] == thisClass]
        # extract
        thisBucket = Parallel(verbose=10, n_jobs=n_threads)(delayed(exhaustiveExtractor)(thisSubset[k], subgraphsOrder, isConnected=True) for k in range(len(thisSubset)))
        thisBucket = [item for sublist in thisBucket for item in sublist]
        # collect
        bucket[thisClass] = thisBucket
elif extractStrategy_Granulator == 'samplePaths':                                                # TODO: move to granulationStrategies.py
    bucket = []
    for o in range(1, subgraphsOrder + 1):      # for each order up to the maximum 'subgraphsOrder'
        thisBucket = []
        thisW = int(round(W / subgraphsOrder))
        while len(thisBucket) < thisW:
            graphID = random.randrange(trSet_size)                                          # pick a graph at random
            thisGraph = trSet[graphID][0]                                                   #
            nodeID = random.randrange(len(thisGraph.nodes()))                               # pick a node at random
            # extract
            try:
                this_o = o if o <= thisGraph.order() else thisGraph.order()
                thisSubgraph = sampleStrategy.random_walk_sampling_simple(thisGraph, this_o)
                thisSubgraph = thisGraph.subgraph(thisSubgraph.nodes()).copy()
                thisSubgraph.name = thisGraph.name + '_' + str(time.time())
                thisBucket.append(thisSubgraph)
            except ValueError:
                pass
        bucket = bucket + thisBucket
        print("Finished extracting for order " + str(o))
elif extractStrategy_Granulator == 'stratSamplePaths':                                           # TODO: move to granulationStrategies.py
    bucket = [None] * numClasses
    for thisClass in range(numClasses):
        print("Working on class " + str(thisClass + 1) + " out of " + str(numClasses))
        thisBucket_class = []
        # strip training set patterns belonging to current class
        thisSubset = [item[0] for item in list(trSet.values()) if item[1] == thisClass]
        # find the number of subgraphs for current class
        thisW_class = int(round(classFrequency_trSet[thisClass] * W))
        # for each order up to the maximum 'subgraphsOrder'
        for o in range(1, subgraphsOrder + 1):
            thisBucket_order = []
            thisW_order = int(round(thisW_class / subgraphsOrder))                              # find the number of subgraphs of a given order for current class
            while len(thisBucket_order) != thisW_order:
                graphID = random.randrange(len(thisSubset))                                     # pick a graph at random
                thisGraph = thisSubset[graphID]                                                 #
                nodeID = random.randrange(len(thisGraph.nodes()))                               # pick a node at random
                # extract
                try:
                    this_o = o if o <= thisGraph.order() else thisGraph.order()
                    thisSubgraph = sampleStrategy.random_walk_sampling_simple(thisGraph, this_o)
                    thisSubgraph = thisGraph.subgraph(thisSubgraph.nodes()).copy()
                    thisSubgraph.name = thisGraph.name + '_' + str(time.time())
                    thisBucket_order.append(thisSubgraph)
                except ValueError:
                    pass
            thisBucket_class = thisBucket_class + thisBucket_order
        print("\tExtracted " + str(len(thisBucket_class)) + " subgraphs for current class.")
        # merge with the overall bucket
        bucket[thisClass] = thisBucket_class
elif extractStrategy_Granulator == 'stratSampleCliques':
    bucket = [None] * numClasses
    for thisClass in range(numClasses):
        print("Working on class " + str(thisClass + 1) + " out of " + str(numClasses))
        thisBucket = []
        # strip training set patterns belonging to current class
        thisSubset = [item[0] for item in list(trSet.values()) if item[1] == thisClass]
        # find the number of subgraphs for current class
        thisW = int(round(classFrequency_trSet[thisClass] * W))
        # extract all cliques from patterns belonging to current class
        for j in range(len(thisSubset)):
            thisBucket = thisBucket + cliqueExtractor(thisSubset[j], maximal=True)
        # random selection
        thisW = thisW if thisW < len(thisBucket) else len(thisBucket)   # if the number of cliques is lower than thisW, we take them all
        IDs = random.sample(range(len(thisBucket)), thisW)
        bucket[thisClass] = [thisBucket[j] for j in IDs]
elif extractStrategy_Granulator == 'stratCliques':
    bucket = [None] * numClasses
    for thisClass in range(numClasses):
        print("Working on class " + str(thisClass + 1) + " out of " + str(numClasses))
        thisBucket = []
        # strip training set patterns belonging to current class
        thisSubset = [item[0] for item in list(trSet.values()) if item[1] == thisClass]
        for k in range(len(thisSubset)):
            thisBucket = thisBucket + cliqueExtractor(thisSubset[k], maximal=True)
        bucket[thisClass] = thisBucket
else:
    raise ValueError('Unknown extract strategy for Granulator')

elapsedTime_ExtractGranulator = time.time() - t
print("Elapsed Time [Extractor Granulator]: " + str(elapsedTime_ExtractGranulator) + " seconds.")
print("\n")


""" Extract for Embedder """
t = time.time()
trSet_EXP = copy.deepcopy(trSet)
vsSet_EXP = copy.deepcopy(vsSet)
tsSet_EXP = copy.deepcopy(tsSet)
if extractStrategy_Embedder == 'paths':
    output = Parallel(verbose=10, n_jobs=n_threads)(delayed(exhaustiveExtractor)(trSet[k][0], subgraphsOrder, isConnected=True) for k in sorted(trSet.keys()))      # process training graphs
    for k in sorted(trSet.keys()):
        trSet_EXP[k] = (output[k], trSet_EXP[k][1])
    output = Parallel(verbose=10, n_jobs=n_threads)(delayed(exhaustiveExtractor)(vsSet[k][0], subgraphsOrder, isConnected=True) for k in sorted(vsSet.keys()))      # process validation graphs
    for k in sorted(vsSet.keys()):
        vsSet_EXP[k] = (output[k], vsSet_EXP[k][1])
    output = Parallel(verbose=10, n_jobs=n_threads)(delayed(exhaustiveExtractor)(tsSet[k][0], subgraphsOrder, isConnected=True) for k in sorted(tsSet.keys()))      # process test graphs
    for k in sorted(tsSet.keys()):
        tsSet_EXP[k] = (output[k], tsSet_EXP[k][1])
elif extractStrategy_Embedder == 'cliques':
    output = Parallel(verbose=10, n_jobs=n_threads)(delayed(cliqueExtractor)(trSet[k][0], maximal=True) for k in sorted(trSet.keys()))                              # process training graphs
    for k in sorted(trSet.keys()):
        trSet_EXP[k] = (output[k], trSet_EXP[k][1])
    output = Parallel(verbose=10, n_jobs=n_threads)(delayed(cliqueExtractor)(vsSet[k][0], maximal=True) for k in sorted(vsSet.keys()))                              # process validation graphs
    for k in sorted(vsSet.keys()):
        vsSet_EXP[k] = (output[k], vsSet_EXP[k][1])
    output = Parallel(verbose=10, n_jobs=n_threads)(delayed(cliqueExtractor)(tsSet[k][0], maximal=True) for k in sorted(tsSet.keys()))                              # process test graphs
    for k in sorted(tsSet.keys()):
        tsSet_EXP[k] = (output[k], tsSet_EXP[k][1])
elif extractStrategy_Embedder == 'bfs':
    output = Parallel(verbose=10, n_jobs=n_threads)(delayed(bfsExtractor)(trSet[k][0], subgraphsOrder) for k in sorted(trSet.keys()))                               # process training graphs
    for k in sorted(trSet.keys()):
        trSet_EXP[k] = (output[k], trSet_EXP[k][1])
    output = Parallel(verbose=10, n_jobs=n_threads)(delayed(bfsExtractor)(vsSet[k][0], subgraphsOrder) for k in sorted(vsSet.keys()))                               # process validation graphs
    for k in sorted(vsSet.keys()):
        vsSet_EXP[k] = (output[k], vsSet_EXP[k][1])
    output = Parallel(verbose=10, n_jobs=n_threads)(delayed(bfsExtractor)(tsSet[k][0], subgraphsOrder) for k in sorted(tsSet.keys()))                               # process test graphs
    for k in sorted(tsSet.keys()):
        tsSet_EXP[k] = (output[k], tsSet_EXP[k][1])
elif extractStrategy_Embedder == 'dfs':
    output = Parallel(verbose=10, n_jobs=n_threads)(delayed(dfsExtractor)(trSet[k][0], subgraphsOrder) for k in sorted(trSet.keys()))                               # process training graphs
    for k in sorted(trSet.keys()):
        trSet_EXP[k] = (output[k], trSet_EXP[k][1])
    output = Parallel(verbose=10, n_jobs=n_threads)(delayed(dfsExtractor)(vsSet[k][0], subgraphsOrder) for k in sorted(vsSet.keys()))                               # process validation graphs
    for k in sorted(vsSet.keys()):
        vsSet_EXP[k] = (output[k], vsSet_EXP[k][1])
    output = Parallel(verbose=10, n_jobs=n_threads)(delayed(dfsExtractor)(tsSet[k][0], subgraphsOrder) for k in sorted(tsSet.keys()))                               # process test graphs
    for k in sorted(tsSet.keys()):
        tsSet_EXP[k] = (output[k], tsSet_EXP[k][1])
else:
    raise ValueError('Unknown extract strategy for Embedder')
del output

elapsedTime_ExtractEmbedder = time.time() - t
print("Elapsed Time [Extractor Embedder]: " + str(elapsedTime_ExtractEmbedder) + " seconds.")
print("\n")


""" GA for Alphabet Tuning """
# t = time.time()
#
# # init genetic algorithm
# toolbox_GA1, pop_GA1, CXPB_GA1, MUTPB_GA1, hof_GA1, stats_GA1, lb_GA1, ub_GA1 = setup_GA1(n_threads, extractStrategy_Granulator, numClasses)
# # register fitness function
# toolbox_GA1.register("evaluate", fitnessfunction_GA1, bucket=bucket, trSet_EXP=trSet_EXP, vsSet_EXP=vsSet_EXP, extractStrategy_Granulator=extractStrategy_Granulator, numClasses=numClasses, graphDissimilarity=graphDissimilarity, lb=lb_GA1, ub=ub_GA1, theta_candidates=theta_candidates, epsilon=epsilon)
# # run GA
# pop_GA1, log_GA1 = algorithms.eaSimple(pop_GA1, toolbox_GA1, cxpb=CXPB_GA1, mutpb=MUTPB_GA1, ngen=20, stats=stats_GA1, halloffame=hof_GA1, verbose=True)
# best_GA1 = hof_GA1[0]
# best_GA1 = clipper(best_GA1, lb_GA1, ub_GA1)
# pop_GA1 = [clipper(item, lb_GA1, ub_GA1) for item in pop_GA1]
# print("Best individual is %s, %s" % (best_GA1, best_GA1.fitness.values))

t = time.time()
bounds_GA1, CXPB_GA1, MUTPB_GA1 = setup_GA1_DE(n_threads, extractStrategy_Granulator, numClasses, dataName)
#Genetic optimization class by class parameters
# Binary ensemble of classifier for class
localConcepts={}
# trSetGlobalEmbed_Mat=numpy.zeros((trSet_size,0),dtype=int)
# vsSetGlobalEmbed_Mat=numpy.zeros((vsSet_size,0),dtype=int)

classe=len(trSet)/numClasses   #suppongo che le classi abbiano tutte lo stesso numero di elementi
classe=int(classe)
coppie=list(combinations(range(numClasses),2))
thisSubset_tr={}
thisSubset_vs={}

# for label in range(len(coppie)):
#     ## Force to be samplePaths since we evolve k optimizer for k-classes
#     extractStrategy_Granulator= "samplePaths"
    
#     j=0
#     for k in trSet.keys():
#         if trSet_EXP[k][1]==coppie[label][0] or trSet_EXP[k][1]==coppie[label][1]:
#             thisSubset_tr[j]=trSet_EXP[k]
#             j=j+1
#     j=0
#     for k in vsSet.keys():
#         if vsSet_EXP[k][1]==coppie[label][0] or vsSet_EXP[k][1]==coppie[label][1]:
#             thisSubset_vs[j]=vsSet_EXP[k]
#             j=j+1

#     # TuningResults_GA1=differential_evolution(fitnessfunction_GA1_DE, \
#     #                                          bounds_GA1, \
#     #                                          args=(label,bucket[label], trSet_EXP, vsSet_EXP, extractStrategy_Granulator, graphDissimilarity, theta_candidates, epsilon, dataName),\
#     #                                          maxiter=20, popsize=round(20/len(bounds_GA1)), \
#     #                                          recombination=CXPB_GA1, \
#     #                                          mutation=MUTPB_GA1, \
#     #                                          workers=n_threads, polish=False, updating='deferred', disp=True)
#     print('\ncoppie: ',coppie[label],'\n')
#     if coppie[label][0]==0 and coppie[label][1]==(numClasses-1):
#         TuningResults_GA1=differential_evolution(fitnessfunction_GA1_DE, \
#                                               bounds_GA1, \
#                                               args=(coppie[label][1],bucket[coppie[label][1]], thisSubset_tr, thisSubset_vs, extractStrategy_Granulator, graphDissimilarity, theta_candidates, epsilon, dataName),\
#                                               maxiter=2, popsize=round(20/len(bounds_GA1)), \
#                                               recombination=CXPB_GA1, \
#                                               mutation=MUTPB_GA1, \
#                                               workers=n_threads, polish=False, updating='deferred', disp=True)
#         best_GA1 = TuningResults_GA1.x
#         """ Embedding with best alphabet """
#         eta = best_GA1[0]
#         tau_f = best_GA1[1]
#         Q = round(best_GA1[2])
        
#         # parsing of additional weights, depending on the dataset
#         if dataName == 'GREC':
#             graphDissimilarity._vParam1 = best_GA1[9]
#             graphDissimilarity._eParam1 = best_GA1[10]
#             graphDissimilarity._eParam2 = best_GA1[11]
#             graphDissimilarity._eParam3 = best_GA1[12]
#             graphDissimilarity._eParam4 = best_GA1[13]
#         if dataName == 'PROTEIN':
#             graphDissimilarity._vParam1 = best_GA1[9]
#             graphDissimilarity._eParam1 = best_GA1[10]
#             graphDissimilarity._eParam2 = best_GA1[11]
#             graphDissimilarity._eParam3 = best_GA1[12]
#             graphDissimilarity._eParam4 = best_GA1[13]
#             graphDissimilarity._eParam5 = best_GA1[14]
        
#         Diss = GED(graphDissimilarity.nodeDissimilarity, graphDissimilarity.edgeDissimilarity)
#         Diss.nodesParam['sub'] = best_GA1[3]
#         Diss.nodesParam['ins'] = best_GA1[4]
#         Diss.nodesParam['del'] = best_GA1[5]
#         Diss.edgesParam['sub'] = best_GA1[6]
#         Diss.edgesParam['ins'] = best_GA1[7]
#         Diss.edgesParam['del'] = best_GA1[8]
        
#         #setting bucket[label] when sampletPaths is "on" is a kind of stratified granulation
#         # if extractStrategy_Granulator in ['samplePaths', 'paths', 'cliques']:
#         #     ALPHABET = ensembleGranulator(bucket[label], Diss.BMF, Q, eta, tau_f, theta_candidates, epsilon, n_jobs=n_threads)
#         # elif extractStrategy_Granulator in ['stratSamplePaths', 'stratSampleCliques']:
#         #     ALPHABET = ensembleStratifiedGranulator(bucket, Diss.BMF, Q, eta, tau_f, numClasses, theta_candidates, epsilon, n_jobs=n_threads)
#         if extractStrategy_Granulator in ['samplePaths', 'paths', 'cliques']:
#             ALPHABET = ensembleGranulator(bucket[coppie[label][1]], Diss.BMF, Q, eta, tau_f, theta_candidates, epsilon, n_jobs=n_threads)
#         elif extractStrategy_Granulator in ['stratSamplePaths', 'stratSampleCliques']:
#             ALPHABET = ensembleStratifiedGranulator(bucket, Diss.BMF, Q, eta, tau_f, numClasses, theta_candidates, epsilon, n_jobs=n_threads)
        
#         alphabet, tau_k = zip(*ALPHABET)
#         trSet_EMB_InstanceMatrix, trSet_EMB_LabelVector = symbolicHistogramsEmbedder(thisSubset_tr, alphabet, tau_k, Diss.BMF, n_jobs=n_threads)
#         vsSet_EMB_InstanceMatrix, vsSet_EMB_LabelVector = symbolicHistogramsEmbedder(thisSubset_vs, alphabet, tau_k, Diss.BMF, n_jobs=n_threads)
#         # trSet_EMB_InstanceMatrix, trSet_EMB_LabelVector = symbolicHistogramsEmbedder(trSet_EXP, alphabet, tau_k, Diss.BMF, n_jobs=n_threads)
#         # vsSet_EMB_InstanceMatrix, vsSet_EMB_LabelVector = symbolicHistogramsEmbedder(vsSet_EXP, alphabet, tau_k, Diss.BMF, n_jobs=n_threads)
        
#         # trSetGlobalEmbed_Mat=numpy.concatenate((trSetGlobalEmbed_Mat,trSet_EMB_InstanceMatrix),axis=1)
#         # vsSetGlobalEmbed_Mat=numpy.concatenate((vsSetGlobalEmbed_Mat,vsSet_EMB_InstanceMatrix),axis=1)
        
#         # Deep copy dissimilarity measure and alphabet into a dict{class label: (Diss_ithClass,Alphabet_ithClass)}
#         # __deepcopy__ operator will recursely copy everything needed to the object to exist,
#         #including derived functions from other classes with relatives parameters.
#         #Consequently this objects are no more binded to referenced objects and parameters for derived classes can be no longer changed
#         # localDissM = copy.deepcopy(Diss)
#         # localAlphabet= copy.deepcopy(ALPHABET)
#         #localConcepts[label] = (copy.deepcopy(Diss), alphabet, tau_k, trSet_EMB_InstanceMatrix, vsSet_EMB_InstanceMatrix)
#         localConcepts[label] = (copy.deepcopy(Diss), alphabet, tau_k, trSet_EMB_InstanceMatrix, vsSet_EMB_InstanceMatrix)

#     else:
#         TuningResults_GA1=differential_evolution(fitnessfunction_GA1_DE, \
#                                               bounds_GA1, \
#                                               args=(coppie[label][0],bucket[coppie[label][0]], thisSubset_tr, thisSubset_vs, extractStrategy_Granulator, graphDissimilarity, theta_candidates, epsilon, dataName),\
#                                               maxiter=2, popsize=round(20/len(bounds_GA1)), \
#                                               recombination=CXPB_GA1, \
#                                               mutation=MUTPB_GA1, \
#                                               workers=n_threads, polish=False, updating='deferred', disp=True)
#         best_GA1 = TuningResults_GA1.x
#         """ Embedding with best alphabet """
#         eta = best_GA1[0]
#         tau_f = best_GA1[1]
#         Q = round(best_GA1[2])
    
#         # parsing of additional weights, depending on the dataset
#         if dataName == 'GREC':
#             graphDissimilarity._vParam1 = best_GA1[9]
#             graphDissimilarity._eParam1 = best_GA1[10]
#             graphDissimilarity._eParam2 = best_GA1[11]
#             graphDissimilarity._eParam3 = best_GA1[12]
#             graphDissimilarity._eParam4 = best_GA1[13]
#         if dataName == 'PROTEIN':
#             graphDissimilarity._vParam1 = best_GA1[9]
#             graphDissimilarity._eParam1 = best_GA1[10]
#             graphDissimilarity._eParam2 = best_GA1[11]
#             graphDissimilarity._eParam3 = best_GA1[12]
#             graphDissimilarity._eParam4 = best_GA1[13]
#             graphDissimilarity._eParam5 = best_GA1[14]
    
#         Diss = GED(graphDissimilarity.nodeDissimilarity, graphDissimilarity.edgeDissimilarity)
#         Diss.nodesParam['sub'] = best_GA1[3]
#         Diss.nodesParam['ins'] = best_GA1[4]
#         Diss.nodesParam['del'] = best_GA1[5]
#         Diss.edgesParam['sub'] = best_GA1[6]
#         Diss.edgesParam['ins'] = best_GA1[7]
#         Diss.edgesParam['del'] = best_GA1[8]
    
#         #setting bucket[label] when sampletPaths is "on" is a kind of stratified granulation
#         # if extractStrategy_Granulator in ['samplePaths', 'paths', 'cliques']:
#         #     ALPHABET = ensembleGranulator(bucket[label], Diss.BMF, Q, eta, tau_f, theta_candidates, epsilon, n_jobs=n_threads)
#         # elif extractStrategy_Granulator in ['stratSamplePaths', 'stratSampleCliques']:
#         #     ALPHABET = ensembleStratifiedGranulator(bucket, Diss.BMF, Q, eta, tau_f, numClasses, theta_candidates, epsilon, n_jobs=n_threads)
#         if extractStrategy_Granulator in ['samplePaths', 'paths', 'cliques']:
#             ALPHABET = ensembleGranulator(bucket[coppie[label][0]], Diss.BMF, Q, eta, tau_f, theta_candidates, epsilon, n_jobs=n_threads)
#         elif extractStrategy_Granulator in ['stratSamplePaths', 'stratSampleCliques']:
#             ALPHABET = ensembleStratifiedGranulator(bucket, Diss.BMF, Q, eta, tau_f, numClasses, theta_candidates, epsilon, n_jobs=n_threads)
    
#         alphabet, tau_k = zip(*ALPHABET)
#         trSet_EMB_InstanceMatrix, trSet_EMB_LabelVector = symbolicHistogramsEmbedder(thisSubset_tr, alphabet, tau_k, Diss.BMF, n_jobs=n_threads)
#         vsSet_EMB_InstanceMatrix, vsSet_EMB_LabelVector = symbolicHistogramsEmbedder(thisSubset_vs, alphabet, tau_k, Diss.BMF, n_jobs=n_threads)
#         # trSet_EMB_InstanceMatrix, trSet_EMB_LabelVector = symbolicHistogramsEmbedder(trSet_EXP, alphabet, tau_k, Diss.BMF, n_jobs=n_threads)
#         # vsSet_EMB_InstanceMatrix, vsSet_EMB_LabelVector = symbolicHistogramsEmbedder(vsSet_EXP, alphabet, tau_k, Diss.BMF, n_jobs=n_threads)
    
#         # trSetGlobalEmbed_Mat=numpy.concatenate((trSetGlobalEmbed_Mat,trSet_EMB_InstanceMatrix),axis=1)
#         # vsSetGlobalEmbed_Mat=numpy.concatenate((vsSetGlobalEmbed_Mat,vsSet_EMB_InstanceMatrix),axis=1)
    
#         # Deep copy dissimilarity measure and alphabet into a dict{class label: (Diss_ithClass,Alphabet_ithClass)}
#         # __deepcopy__ operator will recursely copy everything needed to the object to exist,
#         #including derived functions from other classes with relatives parameters.
#         #Consequently this objects are no more binded to referenced objects and parameters for derived classes can be no longer changed
#         # localDissM = copy.deepcopy(Diss)
#         # localAlphabet= copy.deepcopy(ALPHABET)
#         localConcepts[label] = (copy.deepcopy(Diss), alphabet, tau_k, trSet_EMB_InstanceMatrix, vsSet_EMB_InstanceMatrix)

bucketCoppie=[None]
for label in range(len(coppie)):   #il numero di coppie è uguale a N(N-1)/2
    extractStrategy_Granulator= "samplePaths"

    bucketCoppie= bucket[coppie[label][0]] + bucket[coppie[label][1]] #alfabeto composto da una coppia di classi
    
    j=0
    for k in trSet.keys():
        if trSet_EXP[k][1]==coppie[label][0] or trSet_EXP[k][1]==coppie[label][1]:
            thisSubset_tr[j]=trSet_EXP[k]
            j=j+1
    j=0
    for k in vsSet.keys():
        if vsSet_EXP[k][1]==coppie[label][0] or vsSet_EXP[k][1]==coppie[label][1]:
            thisSubset_vs[j]=vsSet_EXP[k]
            j=j+1
                
    print('\ncoppie: ',coppie[label],'\n')
    TuningResults_GA1=differential_evolution(fitnessfunction_GA1_DE, \
                                              bounds_GA1, \
                                              args=(coppie[label][0], bucketCoppie, thisSubset_tr, thisSubset_vs, extractStrategy_Granulator, graphDissimilarity, theta_candidates, epsilon, dataName),\
                                              maxiter=20, popsize=round(20/len(bounds_GA1)), \
                                              recombination=CXPB_GA1, \
                                              mutation=MUTPB_GA1, \
                                              workers=n_threads, polish=False, updating='deferred', disp=True)
    best_GA1 = TuningResults_GA1.x
    """ Embedding with best alphabet """
    eta = best_GA1[0]
    tau_f = best_GA1[1]
    Q = round(best_GA1[2])
    
    # parsing of additional weights, depending on the dataset
    if dataName == 'GREC':
        graphDissimilarity._vParam1 = best_GA1[9]
        graphDissimilarity._eParam1 = best_GA1[10]
        graphDissimilarity._eParam2 = best_GA1[11]
        graphDissimilarity._eParam3 = best_GA1[12]
        graphDissimilarity._eParam4 = best_GA1[13]
    if dataName == 'PROTEIN':
        graphDissimilarity._vParam1 = best_GA1[9]
        graphDissimilarity._eParam1 = best_GA1[10]
        graphDissimilarity._eParam2 = best_GA1[11]
        graphDissimilarity._eParam3 = best_GA1[12]
        graphDissimilarity._eParam4 = best_GA1[13]
        graphDissimilarity._eParam5 = best_GA1[14]
    
    Diss = GED(graphDissimilarity.nodeDissimilarity, graphDissimilarity.edgeDissimilarity)
    Diss.nodesParam['sub'] = best_GA1[3]
    Diss.nodesParam['ins'] = best_GA1[4]
    Diss.nodesParam['del'] = best_GA1[5]
    Diss.edgesParam['sub'] = best_GA1[6]
    Diss.edgesParam['ins'] = best_GA1[7]
    Diss.edgesParam['del'] = best_GA1[8]
    
    #setting bucket[label] when sampletPaths is "on" is a kind of stratified granulation
    if extractStrategy_Granulator in ['samplePaths', 'paths', 'cliques']:
        ALPHABET = ensembleGranulator(bucketCoppie, Diss.BMF, Q, eta, tau_f, theta_candidates, epsilon, n_jobs=n_threads)
    elif extractStrategy_Granulator in ['stratSamplePaths', 'stratSampleCliques']:
        ALPHABET = ensembleStratifiedGranulator(bucket, Diss.BMF, Q, eta, tau_f, numClasses, theta_candidates, epsilon, n_jobs=n_threads)
    
    print(best_GA1)
    print("ALPHABET size: {}".format(len(ALPHABET)) )
    if len(ALPHABET)==0:
        print("coppia\t",coppie[label])
        print("eta\t", eta)
        print("tau_f\t",tau_f)
        print("Q\t",Q)
        print("Diss.nodesParam[sub]\t",Diss.nodesParam['sub'])
        print("Diss.nodesParam[ins]\t",Diss.nodesParam['ins'])
        print("Diss.nodesParam[del]\t",Diss.nodesParam['del'])
        print("Diss.edgesParam[sub]\t",Diss.edgesParam['sub'])
        print("Diss.edgesParam[ins]\t",Diss.edgesParam['ins'])
        print("Diss.edgesParam[del]\t",Diss.edgesParam['del'])
        print("theta_candidates\t",theta_candidates)
        print("epsilon\t",epsilon)
        pickle.dump({'Diss.nodesParam[sub]':Diss.nodesParam['sub'], 
                     'Diss.nodesParam[ins]':Diss.nodesParam['ins'],
                     'Diss.nodesParam[del]':Diss.nodesParam['del'],
                     'Diss.edgesParam[sub]':Diss.edgesParam['sub'],
                     'Diss.edgesParam[ins]':Diss.edgesParam['ins'],
                     'Diss.edgesParam[del]':Diss.edgesParam['del'],
                     'Q':Q,
                     'eta':eta,
                     'tau_f':tau_f,
                     'theta_candidates':theta_candidates,
                     'epsilon':epsilon,
                     'bucketCoppie':bucketCoppie},
                     open(dataName + '_' + extractStrategy_Granulator + '_' + coppie[label][0] + '_' + coppie[label][1] + '.pkl','wb'))
        #ALPHABET = ensembleGranulator(bucketCoppie, Diss.BMF, Q, eta, tau_f, theta_candidates, epsilon, n_jobs=n_threads)
        raise('error')
    alphabet, tau_k = zip(*ALPHABET)
    trSet_EMB_InstanceMatrix, trSet_EMB_LabelVector = symbolicHistogramsEmbedder(trSet_EXP, alphabet, tau_k, Diss.BMF, n_jobs=n_threads)
    vsSet_EMB_InstanceMatrix, vsSet_EMB_LabelVector = symbolicHistogramsEmbedder(vsSet_EXP, alphabet, tau_k, Diss.BMF, n_jobs=n_threads)
    
    # trSetGlobalEmbed_Mat=numpy.concatenate((trSetGlobalEmbed_Mat,trSet_EMB_InstanceMatrix),axis=1)
    # vsSetGlobalEmbed_Mat=numpy.concatenate((vsSetGlobalEmbed_Mat,vsSet_EMB_InstanceMatrix),axis=1)
    
    # Deep copy dissimilarity measure and alphabet into a dict{class label: (Diss_ithClass,Alphabet_ithClass)}
    # __deepcopy__ operator will recursely copy everything needed to the object to exist,
    #including derived functions from other classes with relatives parameters.
    #Consequently this objects are no more binded to referenced objects and parameters for derived classes can be no longer changed
    # localDissM = copy.deepcopy(Diss)
    # localAlphabet= copy.deepcopy(ALPHABET)
    localConcepts[label] = (copy.deepcopy(Diss), alphabet, tau_k, trSet_EMB_InstanceMatrix, vsSet_EMB_InstanceMatrix)
   
#ALPHABET, tau_k = zip(*ALPHABET)
#trSet_EMB_InstanceMatrix, trSet_EMB_LabelVector = symbolicHistogramsEmbedder(trSet_EXP, ALPHABET, tau_k, Diss.BMF, n_jobs=n_threads)
#vsSet_EMB_InstanceMatrix, vsSet_EMB_LabelVector = symbolicHistogramsEmbedder(vsSet_EXP, ALPHABET, tau_k, Diss.BMF, n_jobs=n_threads)

elapsedTime_FirstGenetic = time.time() - t
print("Elapsed Time [First GA]: " + str(elapsedTime_FirstGenetic) + " seconds.")
print("\n")

# for label in range(numClasses):
#     localConcepts[label] = list(localConcepts[label])   # so we can change it later for test set
for label in range(len(coppie)):
     localConcepts[label] = list(localConcepts[label])   # so we can change it later for test set

# concatenate instance matrices
trSet_EMB_InstanceMatrix, vsSet_EMB_InstanceMatrix  = [None] * len(coppie), [None] * len(coppie)
for label in range(len(coppie)):
    trSet_EMB_InstanceMatrix[label] = localConcepts[label][3]
    vsSet_EMB_InstanceMatrix[label] = localConcepts[label][4]
trSet_EMB_InstanceMatrix = numpy.concatenate(trSet_EMB_InstanceMatrix, axis=1)
vsSet_EMB_InstanceMatrix = numpy.concatenate(vsSet_EMB_InstanceMatrix, axis=1)
# trSet_EMB_InstanceMatrix, vsSet_EMB_InstanceMatrix  = [None] * numClasses, [None] * numClasses
# for label in range(numClasses):
#     trSet_EMB_InstanceMatrix[label] = localConcepts[label][3]
#     vsSet_EMB_InstanceMatrix[label] = localConcepts[label][4]
# trSet_EMB_InstanceMatrix = numpy.concatenate(trSet_EMB_InstanceMatrix, axis=1)
# vsSet_EMB_InstanceMatrix = numpy.concatenate(vsSet_EMB_InstanceMatrix, axis=1)


# concatenate the alphabet
ALPHABET = []
tau_k = []
for label in range(len(coppie)):
    ALPHABET = ALPHABET + list(localConcepts[label][1])
    tau_k = tau_k + list(localConcepts[label][2])
# for label in range(numClasses):
#     ALPHABET = ALPHABET + list(localConcepts[label][1])
#     tau_k = tau_k + list(localConcepts[label][2])

""" GA for Feature Selection """
# t = time.time()
#
# # init genetic algorithm
# toolbox_GA2, pop_GA2, CXPB_GA2, MUTPB_GA2, hof_GA2, stats_GA2, lb_GA2, ub_GA2 = setup_GA2(len(ALPHABET), n_threads)
# # register the goal / fitness function
# toolbox_GA2.register("evaluate", fitnessfunction_GA2, trSet_EMB_InstanceMatrix=trSet_EMB_InstanceMatrix, vsSet_EMB_InstanceMatrix=vsSet_EMB_InstanceMatrix, trSet_EMB_LabelVector=trSet_EMB_LabelVector, vsSet_EMB_LabelVector=vsSet_EMB_LabelVector, lb=lb_GA2, ub=ub_GA2)
# # run GA
# pop_GA2, log_GA2 = algorithms.eaSimple(pop_GA2, toolbox_GA2, cxpb=CXPB_GA2, mutpb=MUTPB_GA2, ngen=20, stats=stats_GA2, halloffame=hof_GA2, verbose=True)
# best_GA2 = hof_GA2[0]
# best_GA2 = clipper(best_GA2, lb_GA2, ub_GA2)
# pop_GA2 = [clipper(item, lb_GA2, ub_GA2) for item in pop_GA2]
# print("Best individual is %s, %s" % (best_GA2, best_GA2.fitness.values))
#
# elapsedTime_SecondGenetic = time.time() - t
# print("Elapsed Time [Second GA]: " + str(elapsedTime_SecondGenetic) + " seconds.")
# print("\n")

###--Comment for debug
t = time.time()
bounds_GA2, CXPB_GA2, MUTPB_GA2, pop = setup_GA2_DE(len(ALPHABET), n_threads)
TuningResults_GA2 = differential_evolution(fitnessfunction_GA2_DE, bounds_GA2, args=(trSet_EMB_InstanceMatrix, vsSet_EMB_InstanceMatrix, trSet_EMB_LabelVector, vsSet_EMB_LabelVector), maxiter=20, init=pop, recombination=CXPB_GA2, mutation=MUTPB_GA2, workers=n_threads, polish=False, updating='deferred', disp=True)
best_GA2 = [round(i) for i in TuningResults_GA2.x]
elapsedTime_SecondGenetic = time.time() - t
###---

""" Test Set """
t = time.time()

tsSetGlobalEmbed_Mat=numpy.zeros((tsSet_size,0),dtype=int)
for label in range(len(coppie)):
    classDiss = localConcepts[label][0]
    classAlphabet = localConcepts[label][1]
    classTau_k = localConcepts[label][2]
    tsSet_EMB_InstanceMatrix, tsSet_EMB_LabelVector = symbolicHistogramsEmbedder(tsSet_EXP, classAlphabet, classTau_k, classDiss.BMF, n_jobs=n_threads)
    localConcepts[label].append(tsSet_EMB_InstanceMatrix)

tsSet_EMB_InstanceMatrix  = [None] * len(coppie)
for label in range(len(coppie)):
    tsSet_EMB_InstanceMatrix[label] = localConcepts[label][-1]
tsSet_EMB_InstanceMatrix = numpy.concatenate(tsSet_EMB_InstanceMatrix, axis=1)
# for label in range(numClasses):
#     classDiss = localConcepts[label][0]
#     classAlphabet = localConcepts[label][1]
#     classTau_k = localConcepts[label][2]
#     tsSet_EMB_InstanceMatrix, tsSet_EMB_LabelVector = symbolicHistogramsEmbedder(tsSet_EXP, classAlphabet, classTau_k, classDiss.BMF, n_jobs=n_threads)
#     localConcepts[label].append(tsSet_EMB_InstanceMatrix)

# tsSet_EMB_InstanceMatrix  = [None] * numClasses
# for label in range(numClasses):
#     tsSet_EMB_InstanceMatrix[label] = localConcepts[label][-1]
# tsSet_EMB_InstanceMatrix = numpy.concatenate(tsSet_EMB_InstanceMatrix, axis=1)

trSet_EMB_InstanceMatrix_shrink = trSet_EMB_InstanceMatrix[:, numpy.array(best_GA2, dtype=bool)]
vsSet_EMB_InstanceMatrix_shrink = vsSet_EMB_InstanceMatrix[:, numpy.array(best_GA2, dtype=bool)]
tsSet_EMB_InstanceMatrix_shrink = tsSet_EMB_InstanceMatrix[:, numpy.array(best_GA2, dtype=bool)]


KNN = KNeighborsClassifier(n_neighbors=5)

#KNN.fit(trSet_EMB_InstanceMatrix_shrink,trSet_EMB_LabelVector)
#predicted_tsSet = KNN.predict(tsSet_EMB_InstanceMatrix_shrink)
#accuracy_tsSet = accuracy_score(tsSet_EMB_LabelVector, predicted_tsSet)

#Moving to ensemble structure on test phase
KNNensemble=OneVsRestClassifier(KNN).fit(trSet_EMB_InstanceMatrix_shrink,trSet_EMB_LabelVector)
predicted_tsSet = KNNensemble.predict(tsSet_EMB_InstanceMatrix_shrink)
accuracy_tsSet = accuracy_score(tsSet_EMB_LabelVector, predicted_tsSet)

elapsedTime_TestPhase = time.time() - t

predicted_trSet = KNNensemble.predict(trSet_EMB_InstanceMatrix_shrink)
predicted_vsSet = KNNensemble.predict(vsSet_EMB_InstanceMatrix_shrink)
accuracy_trSet = accuracy_score(trSet_EMB_LabelVector, predicted_trSet)
accuracy_vlSet = accuracy_score(vsSet_EMB_LabelVector, predicted_vsSet)

print("Accuracy Training Set: " + str(accuracy_trSet))
print("Accuracy Validation Set: " + str(accuracy_vlSet))
print("Accuracy Test Set: " + str(accuracy_tsSet))
print("\n")


""" Save """
pickle.dump({'ALPHABET': ALPHABET,
    'CXPB_GA1': CXPB_GA1,
    'CXPB_GA2': CXPB_GA2,
    'KNN': KNN,
    'MUTPB_GA1': MUTPB_GA1,
    'MUTPB_GA2': MUTPB_GA2,
    'W': W,
    'accuracy_trSet': accuracy_trSet,
    'accuracy_tsSet': accuracy_tsSet,
    'accuracy_vlSet': accuracy_vlSet,
    'best_GA1_individual': best_GA1,
    'best_GA2_individual': best_GA2,
    'bucket': bucket,
    'classMapper': classMapper,
    'elapsedTime_ExtractEmbedder': elapsedTime_ExtractEmbedder,
    'elapsedTime_ExtractGranulator': elapsedTime_ExtractGranulator,
    'elapsedTime_FirstGenetic': elapsedTime_FirstGenetic,
    'elapsedTime_Loader': elapsedTime_Loader,
    'elapsedTime_SecondGenetic': elapsedTime_SecondGenetic,
    'elapsedTime_TestPhase': elapsedTime_TestPhase,
    'extractStrategy_Embedder': extractStrategy_Embedder,
    'extractStrategy_Granulator': extractStrategy_Granulator,
    'log_GA1': TuningResults_GA1,
    'log_GA2': TuningResults_GA2,
    'numClasses': numClasses,
    'predicted_trSet': predicted_trSet,
    'predicted_tsSet': predicted_tsSet,
    'predicted_vsSet': predicted_vsSet,
    'subgraphsOrder': subgraphsOrder,
    'tau_k': tau_k,
    'theta_candidates': theta_candidates,
    'localConcepts': localConcepts,
    'trSet': trSet,
    'trSet_EMB_InstanceMatrix': trSet_EMB_InstanceMatrix,
    'trSet_EMB_InstanceMatrix_shrink': trSet_EMB_InstanceMatrix_shrink,
    'trSet_EMB_LabelVector': trSet_EMB_LabelVector,
    'trSet_EXP': trSet_EXP,
    'tsSet': tsSet,
    'tsSet_EMB_InstanceMatrix': tsSet_EMB_InstanceMatrix,
    'tsSet_EMB_InstanceMatrix_shrink': tsSet_EMB_InstanceMatrix_shrink,
    'tsSet_EMB_LabelVector': tsSet_EMB_LabelVector,
    'tsSet_EXP': tsSet_EXP,
    'vsSet': vsSet,
    'vsSet_EMB_InstanceMatrix': vsSet_EMB_InstanceMatrix,
    'vsSet_EMB_InstanceMatrix_shrink': vsSet_EMB_InstanceMatrix_shrink,
    'vsSet_EMB_LabelVector': vsSet_EMB_LabelVector,
    'vsSet_EXP': vsSet_EXP,
    'Workstation': os.uname()[1]}, open(dataName + '_' + extractStrategy_Granulator + '_' + runID + '.pkl', 'wb'))

# pickle.dump({'ALPHABET': ALPHABET,
#     'CXPB_GA1': CXPB_GA1,
#     'CXPB_GA2': CXPB_GA2,
#     'KNN': KNN,
#     'MUTPB_GA1': MUTPB_GA1,
#     'MUTPB_GA2': MUTPB_GA2,
#     'W': W,
#     'accuracy_trSet': accuracy_trSet,
#     'accuracy_tsSet': accuracy_tsSet,
#     'accuracy_vlSet': accuracy_vlSet,
#     'best_GA1_individual': list(best_GA1),
#     'best_GA1_fitness': best_GA1.fitness.values,
#     'best_GA2_individual': list(best_GA2),
#     'best_GA2_fitness': best_GA2.fitness.values,
#     'bucket': bucket,
#     'classMapper': classMapper,
#     'elapsedTime_ExtractEmbedder': elapsedTime_ExtractEmbedder,
#     'elapsedTime_ExtractGranulator': elapsedTime_ExtractGranulator,
#     'elapsedTime_FirstGenetic': elapsedTime_FirstGenetic,
#     'elapsedTime_Loader': elapsedTime_Loader,
#     'elapsedTime_SecondGenetic': elapsedTime_SecondGenetic,
#     'elapsedTime_TestPhase': elapsedTime_TestPhase,
#     'extractStrategy_Embedder': extractStrategy_Embedder,
#     'extractStrategy_Granulator': extractStrategy_Granulator,
#     'log_GA1': log_GA1,
#     'log_GA2': log_GA2,
#     'numClasses': numClasses,
#     'finalPop_GA1_individual': [list(item) for item in pop_GA1],
#     'finalPop_GA1_fitness': [pop_GA1[i].fitness.values for i in range(len(pop_GA1))],
#     'finalPop_GA2_individual': [list(item) for item in pop_GA2],
#     'finalPop_GA2_fitness': [pop_GA2[i].fitness.values for i in range(len(pop_GA2))],
#     'predicted_trSet': predicted_trSet,
#     'predicted_tsSet': predicted_tsSet,
#     'predicted_vsSet': predicted_vsSet,
#     'subgraphsOrder': subgraphsOrder,
#     'tau_k': tau_k,
#     'theta_candidates': theta_candidates,
#     'trSet': trSet,
#     'trSet_EMB_InstanceMatrix': trSet_EMB_InstanceMatrix,
#     'trSet_EMB_InstanceMatrix_shrink': trSet_EMB_InstanceMatrix_shrink,
#     'trSet_EMB_LabelVector': trSet_EMB_LabelVector,
#     'trSet_EXP': trSet_EXP,
#     'tsSet': tsSet,
#     'tsSet_EMB_InstanceMatrix': tsSet_EMB_InstanceMatrix,
#     'tsSet_EMB_InstanceMatrix_shrink': tsSet_EMB_InstanceMatrix_shrink,
#     'tsSet_EMB_LabelVector': tsSet_EMB_LabelVector,
#     'tsSet_EXP': tsSet_EXP,
#     'vsSet': vsSet,
#     'vsSet_EMB_InstanceMatrix': vsSet_EMB_InstanceMatrix,
#     'vsSet_EMB_InstanceMatrix_shrink': vsSet_EMB_InstanceMatrix_shrink,
#     'vsSet_EMB_LabelVector': vsSet_EMB_LabelVector,
#     'vsSet_EXP': vsSet_EXP,
#     'Workstation': os.uname()[1]}, open(dataName + '_' + extractStrategy_Granulator + '_' + runID + '.pkl', 'wb'))
