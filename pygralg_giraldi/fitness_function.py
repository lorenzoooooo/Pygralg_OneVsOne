# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 14:22:49 2021

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

sampleStrategy = Graph_Sampling.SRW_RWF_ISRW()                  # declare the sampling strategy (ignored for exhaustive- and clique-based strategies)
subgraphsOrder = 5                                              # max subgraphs order (only if cliques are NOT involved)
n_threads = 1
delimiters = "_", "."
theta_candidates = BSP(0, 1, 0.1)
epsilon=1.1
W=797
extractStrategy_Granulator='stratPaths'
dataName='Letter1'

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
normFactor = normalize(trSet, vsSet, tsSet)
graphDissimilarity._VertexDissWeights = normFactor


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
'''stratPaths'''
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

'''Extractor for Embedder'''
'''Paths'''
# trSet_EXP = copy.deepcopy(trSet)
# vsSet_EXP = copy.deepcopy(vsSet)
# tsSet_EXP = copy.deepcopy(tsSet)
# output = Parallel(verbose=10, n_jobs=n_threads)(delayed(exhaustiveExtractor)(trSet[k][0], subgraphsOrder, isConnected=True) for k in sorted(trSet.keys()))      # process training graphs
# for k in sorted(trSet.keys()):
#     trSet_EXP[k] = (output[k], trSet_EXP[k][1])
# output = Parallel(verbose=10, n_jobs=n_threads)(delayed(exhaustiveExtractor)(vsSet[k][0], subgraphsOrder, isConnected=True) for k in sorted(vsSet.keys()))      # process validation graphs
# for k in sorted(vsSet.keys()):
#     vsSet_EXP[k] = (output[k], vsSet_EXP[k][1])
# output = Parallel(verbose=10, n_jobs=n_threads)(delayed(exhaustiveExtractor)(tsSet[k][0], subgraphsOrder, isConnected=True) for k in sorted(tsSet.keys()))      # process test graphs
# for k in sorted(tsSet.keys()):
#     tsSet_EXP[k] = (output[k], tsSet_EXP[k][1])
    
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

# coopie 1-9
extractStrategy_Granulator= "samplePaths"
bucketCoppie= bucket[1] + bucket[9] #alfabeto composto da una coppia di classi
# j=0
# for k in trSet.keys():
#     if trSet_EXP[k][1]==1 or trSet_EXP[k][1]==9:
#         thisSubset_tr[j]=trSet_EXP[k]
#         j=j+1
# j=0
# for k in vsSet.keys():
#     if vsSet_EXP[k][1]==1 or vsSet_EXP[k][1]==9:
#         thisSubset_vs[j]=vsSet_EXP[k]
#         j=j+1
eta=0.7303868359872551                               #best_GA1[0]
tau_f=0.04248750722885508                            #best_GA1[1]
Q=4                                                  #best_GA1[2]
# GED setup
Diss = GED(graphDissimilarity.nodeDissimilarity, graphDissimilarity.edgeDissimilarity)
Diss.nodesParam['sub'] = 0.9918734024806518          #best_GA1[3]
Diss.nodesParam['ins'] = 0.34797857594874326         #best_GA1[4]
Diss.nodesParam['del'] = 0.4532498430956843          #best_GA1[5]
Diss.edgesParam['sub'] = 0.3508015365254712          #best_GA1[6]
Diss.edgesParam['ins'] = 0.6847256434510226          #best_GA1[7]
Diss.edgesParam['del'] = 0.4815671531792641          #best_GA1[8]


# Set useful parameters
alpha = 0.9


ALPHABET = ensembleGranulator(bucketCoppie, Diss.BMF, Q, eta, tau_f, theta_candidates, epsilon)
bucketSize = len(bucket)


# Prior exit if alphabet is empty
if ALPHABET == []:
    print("empty alphabet\n")

ALPHABET, tau_k = zip(*ALPHABET)

# # Embedder
# trSet_EMB_InstanceMatrix, trSet_EMB_LabelVector = symbolicHistogramsEmbedder(thisSubset_tr, ALPHABET, tau_k, Diss.BMF)
# vsSet_EMB_InstanceMatrix, vsSet_EMB_LabelVector = symbolicHistogramsEmbedder(thisSubset_vs, ALPHABET, tau_k, Diss.BMF)

# # Class relabelling where target class is 1
# trSet_EMB_LabelVector = (trSet_EMB_LabelVector == 1).astype(int)
# vsSet_EMB_LabelVector = (vsSet_EMB_LabelVector == 1).astype(int)

# # Classifier
# KNN = KNeighborsClassifier(n_neighbors=5)
# KNN.fit(trSet_EMB_InstanceMatrix, trSet_EMB_LabelVector)
# predicted_vsSet = KNN.predict(vsSet_EMB_InstanceMatrix)

# #Move to informedness
# """ From sci-kit lib confusion matrix return C as follow:
# [...]Thus in binary classification, the count of true negatives is
# :math:`C_{0,0}`, false negatives is :math:`C_{1,0}`, true positives is
# :math:`C_{1,1}` and false positives is :math:`C_{0,1}`.
# """    
# tn, fp, fn, tp = confusion_matrix(vsSet_EMB_LabelVector, predicted_vsSet).ravel()
# sensitivity = tp / (tp + fn)
# specificity = tn / (tn + fp)
# J = sensitivity + specificity - 1
# J = (J + 1) / 2
# error_rate = 1 - J

# # accuracy = accuracy_score(vsSet_EMB_LabelVector, predicted_vsSet)
# # error_rate = 1 - accuracy
    
# fitness = alpha * error_rate + (1 - alpha) * (len(ALPHABET) / bucketSize)   # add a small term in order to prefer small alphabets upon (pretty much) the same accuracy
# # print("Parameters: " + str([round(i, 2) for i in genetic_code]) + "\tAccuracy " + str(accuracy) + "\tAlphabet: " + str(len(ALPHABET)))
# print("Parameters: " + str([round(i, 2) for i in genetic_code]) + "\tInformedness " + str(J) + "\tAlphabet: " + str(len(ALPHABET)))