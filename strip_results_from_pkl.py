import pickle
import numpy

# Load pkl files
RESULTS = [None] * 3    # we have 5 runs
RESULTS[0] = pickle.load(open('results path subsampling/AIDS_stratSamplePaths_30_1.pkl', 'rb'))
RESULTS[1] = pickle.load(open('results path subsampling/AIDS_stratSamplePaths_30_2.pkl', 'rb'))
RESULTS[2] = pickle.load(open('results path subsampling/AIDS_stratSamplePaths_30_3.pkl', 'rb'))
# RESULTS[3] = pickle.load(open('results clique subsampling/AIDS_exhaustive_4.pkl', 'rb'))
# RESULTS[4] = pickle.load(open('results clique subsampling/AIDS_exhaustive_5.pkl', 'rb'))

# Strip accuracy on test set
ACC = [item['accuracy_tsSet'] for item in RESULTS]

# Strip alphabet size before feature selection
ALP_before = [len(item['best_GA2_individual']) for item in RESULTS]

# Strip alphabet size after feature selection
ALP_after = [sum(item['best_GA2_individual']) for item in RESULTS]

# Strip running time (in minutes) for first GA
time_firstGA = [item['elapsedTime_FirstGenetic'] / 60 for item in RESULTS]

# Strip running time (in minutes) for second GA
time_secondGA = [item['elapsedTime_SecondGenetic'] / 60 for item in RESULTS]

# Strip running time (in minutes) for test phase
time_test = [item['elapsedTime_TestPhase'] / 60 for item in RESULTS]

print('Accuracy on Test Set: ' + str(round(numpy.mean(ACC), 2)) + " ± " + str(round(numpy.std(ACC), 2)))
print('Alphabet Size: ' + str(round(numpy.mean(ALP_before), 2)) + " ± " + str(round(numpy.std(ALP_before), 2)))
print('Selected Features: ' + str(round(numpy.mean(ALP_after), 2)) + " ± " + str(round(numpy.std(ALP_after), 2)))
print('First GA time: ' + str(round(numpy.mean(time_firstGA), 2)) + " ± " + str(round(numpy.std(time_firstGA), 2)))
print('Second GA time: ' + str(round(numpy.mean(time_secondGA), 2)) + " ± " + str(round(numpy.std(time_secondGA), 2)))
print('Test phase time: ' + str(round(numpy.mean(time_test), 2)) + " ± " + str(round(numpy.std(time_test), 2)))


# # strip selected symbols
# run0 = numpy.where(numpy.array(RESULTS[0]['best_GA2_individual'])!=0)[0].tolist()
# run1 = numpy.where(numpy.array(RESULTS[1]['best_GA2_individual'])!=0)[0].tolist()
# run2 = numpy.where(numpy.array(RESULTS[2]['best_GA2_individual'])!=0)[0].tolist()
# # run3 = numpy.where(numpy.array(RESULTS[3]['best_GA2_individual'])!=0)[0].tolist()
# # run4 = numpy.where(numpy.array(RESULTS[4]['best_GA2_individual'])!=0)[0].tolist()
# 
# run0 = [RESULTS[0]['ALPHABET'][i] for i in run0]
# run1 = [RESULTS[1]['ALPHABET'][i] for i in run1]
# run2 = [RESULTS[2]['ALPHABET'][i] for i in run2]
# # run3 = [RESULTS[3]['ALPHABET'][i] for i in run3]
# # run4 = [RESULTS[4]['ALPHABET'][i] for i in run4]
# 
# print('## Run 1')
# for i in range(len(run0)):
#     print('# Subgraph ' + str(i+1))
#     print(run0[i].nodes(data=True))
#     print(run0[i].edges(data=True))
# print('\n## Run 2')
# for i in range(len(run1)):
#     print('# Subgraph ' + str(i+1))
#     print(run1[i].nodes(data=True))
#     print(run1[i].edges(data=True))
# print('\n## Run 3')
# for i in range(len(run2)):
#     print('# Subgraph ' + str(i+1))
#     print(run2[i].nodes(data=True))
#     print(run2[i].edges(data=True))
# # print('\n## Run 4')
# # for i in range(len(run3)):
# #     print('# Subgraph ' + str(i+1))
# #     print(run3[i].nodes(data=True))
# #     print(run3[i].edges(data=True))
# # print('\n## Run 5')
# # for i in range(len(run4)):
# #     print('# Subgraph ' + str(i+1))
# #     print(run4[i].nodes(data=True))
# #     print(run4[i].edges(data=True))
