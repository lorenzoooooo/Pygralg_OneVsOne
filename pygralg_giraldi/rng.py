# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 10:58:56 2021

@author: Utente
"""

import numpy
import itertools
import random
# import copy
from scipy.sparse import coo_matrix, dok_matrix

poolSize = 20
random.seed(1)
id_pattern1, id_pattern2 = random.sample(range(0, poolSize), 2)
print('\nid_pattern1\t',id_pattern1,'\nid_pattern2\t',id_pattern2,'\n')

# bucketCoppie=[None]

# extractStrategy_Granulator= "samplePaths"

# bucketCoppie= bucket[1] + bucket[9] #alfabeto composto da una coppia di classi

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
# #fitnessfunction_GA1_DE()            
# TuningResults_GA1=differential_evolution(fitnessfunction_GA1_DE, \
#                                           bounds_GA1, \
#                                           args=(1, bucketCoppie, thisSubset_tr, thisSubset_vs, extractStrategy_Granulator, graphDissimilarity, theta_candidates, epsilon, dataName),\
#                                           maxiter=2, popsize=round(20/len(bounds_GA1)), \
#                                           recombination=CXPB_GA1, \
#                                           mutation=MUTPB_GA1, \
#                                           workers=n_threads, polish=False, updating='deferred', disp=True)
# best_GA1 = TuningResults_GA1.x
# """ Embedding with best alphabet """
# eta = best_GA1[0]
# tau_f = best_GA1[1]
# Q = round(best_GA1[2])

# # parsing of additional weights, depending on the dataset
# if dataName == 'GREC':
#     graphDissimilarity._vParam1 = best_GA1[9]
#     graphDissimilarity._eParam1 = best_GA1[10]
#     graphDissimilarity._eParam2 = best_GA1[11]
#     graphDissimilarity._eParam3 = best_GA1[12]
#     graphDissimilarity._eParam4 = best_GA1[13]
# if dataName == 'PROTEIN':
#     graphDissimilarity._vParam1 = best_GA1[9]
#     graphDissimilarity._eParam1 = best_GA1[10]
#     graphDissimilarity._eParam2 = best_GA1[11]
#     graphDissimilarity._eParam3 = best_GA1[12]
#     graphDissimilarity._eParam4 = best_GA1[13]
#     graphDissimilarity._eParam5 = best_GA1[14]

# Diss = GED(graphDissimilarity.nodeDissimilarity, graphDissimilarity.edgeDissimilarity)
# Diss.nodesParam['sub'] = best_GA1[3]
# Diss.nodesParam['ins'] = best_GA1[4]
# Diss.nodesParam['del'] = best_GA1[5]
# Diss.edgesParam['sub'] = best_GA1[6]
# Diss.edgesParam['ins'] = best_GA1[7]
# Diss.edgesParam['del'] = best_GA1[8]

# #setting bucket[label] when sampletPaths is "on" is a kind of stratified granulation
# if extractStrategy_Granulator in ['samplePaths', 'paths', 'cliques']:
#     ALPHABET = ensembleGranulator(bucketCoppie, Diss.BMF, Q, eta, tau_f, theta_candidates, epsilon, n_jobs=n_threads)
# elif extractStrategy_Granulator in ['stratSamplePaths', 'stratSampleCliques']:
#     ALPHABET = ensembleStratifiedGranulator(bucket, Diss.BMF, Q, eta, tau_f, numClasses, theta_candidates, epsilon, n_jobs=n_threads)
