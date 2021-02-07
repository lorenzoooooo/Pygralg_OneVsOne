# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 20:07:26 2021

@author: Utente
"""

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

""" Set some useful parameters """                              # NOTE: some of them can be moved to command line
sampleStrategy = Graph_Sampling.SRW_RWF_ISRW()                  # declare the sampling strategy (ignored for exhaustive- and clique-based strategies)
subgraphsOrder = 5                                              # max subgraphs order (only if cliques are NOT involved)
theta_candidates = BSP(0, 1, 0.1)                               # list of theta candidates for BSAS
epsilon = 1.1                                                   # tolerance value in symbols recognition
#n_threads = -1                                                  # number of threads for parallel execution
n_threads = 1
delimiters = "_", "."                                           # Name file contains id and label


""" Set metric and problem """
from util.normalizeLetter import normalize
from GraphTypes import Letter

graphDissimilarity = Letter.LETTERdiss()
parserFunction = Letter.parser

trDir = "C:/Users/Utente/Documents/Lorenzo/materiale_didattico_sapienza/lezioni/PR/pyGRALG_LML-master/dataset_IAM/Letter1/Training/"                                                 # paths DIETrack1, 2 and 3
vsDir = "C:/Users/Utente/Documents/Lorenzo/materiale_didattico_sapienza/lezioni/PR/pyGRALG_LML-master/dataset_IAM/Letter1/Validation/"
tsDir = "C:/Users/Utente/Documents/Lorenzo/materiale_didattico_sapienza/lezioni/PR/pyGRALG_LML-master/dataset_IAM/Letter1/Test/"
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
# if dataName == 'AIDS':
#     normFactorVertex = normalize(trSet, vsSet, tsSet)
#     graphDissimilarity._VertexDissWeights = normFactorVertex
# elif dataName in ['Letter1', 'Letter2', 'Letter3']:
normFactor = normalize(trSet, vsSet, tsSet)
graphDissimilarity._VertexDissWeights = normFactor
# elif dataName == 'GREC':
#     normFactorVertex, normFactorEdge = normalize(trSet, vsSet, tsSet)
#     graphDissimilarity._VertexDissWeights = normFactorVertex
#     graphDissimilarity._EdgeDissWeights = normFactorEdge
# elif dataName == 'PROTEIN':
#     normFactor = normalize(trSet, vsSet, tsSet)
#     graphDissimilarity._EdgeDissWeights = normFactor

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
    tsSet[k] = (thisGraph, tsSet[k][1])     

classFrequency_trSet = dict(collections.Counter([item[1] for item in list(trSet.values())]))    # let us evaluate
classFrequency_vsSet = dict(collections.Counter([item[1] for item in list(vsSet.values())]))    # the frequency
classFrequency_tsSet = dict(collections.Counter([item[1] for item in list(tsSet.values())]))    # of each class
for k in classFrequency_trSet.keys():                                                           # within each set
    classFrequency_trSet[k] = classFrequency_trSet[k] / trSet_size                              #
for k in classFrequency_vsSet.keys():                                                           # this is useful
    classFrequency_vsSet[k] = classFrequency_vsSet[k] / vsSet_size                              # for stratified sampling
for k in classFrequency_tsSet.keys():                                                           # strategies
    classFrequency_tsSet[k] = classFrequency_tsSet[k] / tsSet_size                                                         #

print("Loaded " + str(trSet_size) + " graphs for Training Set.")
print("Loaded " + str(vsSet_size) + " graphs for Validation Set.")
print("Loaded " + str(tsSet_size) + " graphs for Test Set.")
print("\n")

'''Extractor for Granulatore'''
# bucket = [None] * numClasses     #dichiaro array di numClasses elementi
# for thisClass in range(numClasses):
#     print("Working on class " + str(thisClass + 1) + " out of " + str(numClasses))
#     # strip training set patterns belonging to current class
#     thisSubset = [item[0] for item in list(trSet.values()) if item[1] == thisClass]
#     # extract
#     thisBucket = Parallel(verbose=10, n_jobs=n_threads)(delayed(exhaustiveExtractor)(thisSubset[k], subgraphsOrder, isConnected=True) for k in range(len(thisSubset)))
#     thisBucket = [item for sublist in thisBucket for item in sublist]
#     # collect
#     bucket[thisClass] = thisBucket
W=797
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

bounds_GA1, CXPB_GA1, MUTPB_GA1 = setup_GA1_DE(n_threads, 'stratSamplePaths', numClasses, 'Letter1')
#Genetic optimization class by class parameters
# Binary ensemble of classifier for class
localConcepts={}
# trSetGlobalEmbed_Mat=numpy.zeros((trSet_size,0),dtype=int)
# vsSetGlobalEmbed_Mat=numpy.zeros((vsSet_size,0),dtype=int)

classe=len(trSet)/numClasses   #suppongo che le classi abbiano tutte lo stesso numero di elementi
classe=int(classe)
coppie=list(combinations(range(numClasses),2))
#thisSubset_tr={}
#thisSubset_vs={}

# eta=0.7303868359872551                               #best_GA1[0]
# #tau_f=0.78                           #best_GA1[1]
# tau_f=0.04248750722885508                            #best_GA1[1]
# Q=4                                                  #best_GA1[2]
# Diss = GED(graphDissimilarity.nodeDissimilarity, graphDissimilarity.edgeDissimilarity)
# Diss.nodesParam['sub'] = 0.9918734024806518          #best_GA1[3]
# Diss.nodesParam['ins'] = 0.34797857594874326         #best_GA1[4]
# Diss.nodesParam['del'] = 0.4532498430956843          #best_GA1[5]
# Diss.edgesParam['sub'] = 0.3508015365254712          #best_GA1[6]
# Diss.edgesParam['ins'] = 0.6847256434510226          #best_GA1[7]
# Diss.edgesParam['del'] = 0.4815671531792641          #best_GA1[8]
eta=0.3876078260594895                               #best_GA1[0]
tau_f=0.08472613477275137                            #best_GA1[1]
Q=3                                                  #best_GA1[2]
Diss = GED(graphDissimilarity.nodeDissimilarity, graphDissimilarity.edgeDissimilarity)
Diss.nodesParam['sub'] = 0.644916026158982          #best_GA1[3]
Diss.nodesParam['ins'] = 0.763627489641594         #best_GA1[4]
Diss.nodesParam['del'] = 0.29230366544777353          #best_GA1[5]
Diss.edgesParam['sub'] = 0.7071137586809673          #best_GA1[6]
Diss.edgesParam['ins'] = 0.9673618676229165          #best_GA1[7]
Diss.edgesParam['del'] = 0.6446656067467292        #best_GA1[8]

theta_candidates=[0, 1, 0.5, 0.25, 0.125]
epsilon=1.1
coppie=list(combinations(range(numClasses),2))
i=0
a={}
for label in range(len(coppie)):
    print('\ncoppie: ',coppie[label],'\n')
    bucketCoppie= bucket[coppie[label][0]] + bucket[coppie[label][1]]
    ALPHABET = ensembleGranulator(bucketCoppie, Diss.BMF, Q, eta, tau_f, theta_candidates, epsilon, 1)
    print("len(ALPHABET)",len(ALPHABET))
    if len(ALPHABET)==0:
        a[i]=coppie[label]
        i=i+1
    pickle.dump({'classe[label]':a}, open('Letter1' + '.pkl','wb'))
#-dataName Letter1 -runID 1 -extractG stratSamplePaths -extractE paths -W 797
#print("[0.73038684 0.04248751 4.04608225 0.9918734  0.34797858 0.45324984 0.35080154 0.68472564 0.48156715]")