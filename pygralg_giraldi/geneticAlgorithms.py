import multiprocessing
import random
import numpy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from deap import base, creator, tools

from embeddingStrategies import symbolicHistogramsEmbedder
from granulationStrategies import ensembleGranulator, ensembleStratifiedGranulator
from GraphDissimilarities.GED import GED
from util.misc import clipper


def fitnessfunction_GA1(genetic_code, bucket, trSet_EXP, vsSet_EXP, extractStrategy_Granulator, numClasses, graphDissimilarity, lb, ub, theta_candidates, epsilon):
    """ Fitness function for first GA (alphabet tuning). To be used with DEAP.

    Input:
    - genetic_code: Individual object provided by DEAP
    - bucket: list of subgraphs (if not stratified) or list-of-lists of class-specific subgraphs (if stratified) to be clustered
    - trSet_EXP: a dictionary of (expanded) training graphs of the form {id: (list-of-subgraphs, label)}
    - vsSet_EXP: a dictionary of (expanded) validation graphs of the form {id: (list-of-subgraphs, label)}
    - extractStrategy_Granulator: string representing the granulation strategy (to be set in main.py)
    - numClasses: number of classes for the classification problem (useful only for stratified approaches)
    - graphDissimilarity: object endowing the dissimilarity
    - lb: list of chromosomes' lower bound values
    - ub: list of chromosomes' upper bound values
    - theta_candidates: list of theta candidates for BSAS
    - epsilon: tolerance value in symbols recognition
    Output:
    - accuracy: accuracy of the classifier. """

    # Check upper and lower bounds
    genetic_code = clipper(genetic_code, lb, ub)

    # Strip parameters from genetic code
    eta = genetic_code[0]
    tau_f = genetic_code[1]
    Q = round(genetic_code[2])
    pVertexSubs = genetic_code[3]
    pVertexIns = genetic_code[4]
    pVertexDel = genetic_code[5]
    pEdgeSubs = genetic_code[6]
    pEdgeIns = genetic_code[7]
    pEdgeDel = genetic_code[8]

    # Set useful parameters
    alpha = 0.99

    # GED setup
    Diss = GED(nodeDissimilarity, edgeDissimilarity)
    Diss.nodesParam['sub'] = pVertexSubs
    Diss.nodesParam['ins'] = pVertexIns
    Diss.nodesParam['del'] = pVertexDel
    Diss.edgesParam['sub'] = pEdgeSubs
    Diss.edgesParam['ins'] = pEdgeIns
    Diss.edgesParam['del'] = pEdgeDel

    # Granulator
    if extractStrategy_Granulator in ['samplePaths', 'paths', 'cliques']:
        ALPHABET = ensembleGranulator(bucket, Diss.BMF, Q, eta, tau_f, theta_candidates, epsilon)
        bucketSize = len(bucket)
    elif extractStrategy_Granulator in ['stratSamplePaths', 'stratSampleCliques']:
        ALPHABET = ensembleStratifiedGranulator(bucket, Diss.BMF, Q, eta, tau_f, numClasses, theta_candidates, epsilon)
        bucketSize = sum([len(bucket[c]) for c in range(numClasses)])

    # Prior exit if alphabet is empty
    if ALPHABET == []:
        print("Parameters: " + str([round(i, 2) for i in genetic_code]) + "\tEmpty Alphabet")
        return (numpy.finfo(numpy.float64).tiny, )

    ALPHABET, tau_k = zip(*ALPHABET)

    # Embedder
    trSet_EMB_InstanceMatrix, trSet_EMB_LabelVector = symbolicHistogramsEmbedder(trSet_EXP, ALPHABET, tau_k, Diss.BMF)
    vsSet_EMB_InstanceMatrix, vsSet_EMB_LabelVector = symbolicHistogramsEmbedder(vsSet_EXP, ALPHABET, tau_k, Diss.BMF)

    # Classifier
    KNN = KNeighborsClassifier(n_neighbors=5)
    KNN.fit(trSet_EMB_InstanceMatrix, trSet_EMB_LabelVector)
    predicted_vsSet = KNN.predict(vsSet_EMB_InstanceMatrix)
    accuracy = accuracy_score(vsSet_EMB_LabelVector, predicted_vsSet)
    f = alpha * accuracy + (1 - alpha) * (1 - (len(ALPHABET) / bucketSize))     # add a small term in order to prefer small alphabets upon (pretty much) the same accuracy
    print("Parameters: " + str([round(i, 2) for i in genetic_code]) + "\tAccuracy: " + str(accuracy) + "\tAlphabet: " + str(len(ALPHABET)))
    return (f, )


def fitnessfunction_GA2(genetic_code, trSet_EMB_InstanceMatrix, vsSet_EMB_InstanceMatrix, trSet_EMB_LabelVector, vsSet_EMB_LabelVector, lb, ub):
    """ Fitness function for second GA (feature selection). To be used with DEAP.

    Input:
    - genetic_code: Individual object provided by DEAP
    - trSet_EMB_InstanceMatrix: embedded training set with the best alphabet
    - vsSet_EMB_InstanceMatrix: embedded validation set with the best alphabet
    - trSet_EMB_LabelVector: training set labels
    - vsSet_EMB_LabelVector: validation set labels
    - lb: list of chromosomes' lower bound values
    - ub: list of chromosomes' upper bound values.
    Output:
    - f: fitness function of the form [alpha * accuracy + (1-alpha) * mask cost]. """

    # Check upper and lower bounds
    genetic_code = clipper(genetic_code, lb, ub)

    # Strip parameters from genetic code
    mask = [round(i) for i in genetic_code]

    # Set useful parameters
    alpha = 0.99

    # Evalaute mask cost
    mask = numpy.array(mask, dtype=bool)
    selectedRatio = sum(mask) / len(genetic_code)
    unselectedRatio = 1 - selectedRatio

    # Prior exit if no features have been selected
    if selectedRatio == 0:
        return (numpy.finfo(numpy.float64).tiny, )

    # Classifier
    KNN = KNeighborsClassifier(n_neighbors=5)
    KNN.fit(trSet_EMB_InstanceMatrix[:, mask], trSet_EMB_LabelVector)
    predicted_vsSet = KNN.predict(vsSet_EMB_InstanceMatrix[:, mask])
    accuracy = accuracy_score(vsSet_EMB_LabelVector, predicted_vsSet)
    f = alpha * accuracy + (1 - alpha) * unselectedRatio
    print("Selection ratio: " + str(selectedRatio) + "\tAccuracy: " + str(accuracy))
    return (f, )


def setup_GA1(n_threads, extractStrategy_Granulator, numClasses):
    """ Parameters for first GA (alphabet tuning). To be used with DEAP.

    Input:
    - n_threads: number of threads for parallel individual evaluation
    - extractStrategy_Granulator: string representing the granulation strategy (to be set in main.py)
    - numClasses: number of classes for the classification problem (useful only for stratified approaches)
    Output:
    - toolbox: a DEAP toolbox including the genetic algorithm setup
    - pop: the initial population
    - CXPB: crossover probability
    - MUTPB: mutation probability
    - hof: the DEAP hall of fame for saving the global best individual
    - stats: DEAP module for capturing basic statistics as the evolution goes by
    - lb: chromosomes lower bounds
    - ub: chromosomes upper bounds. """

    # register individual type and fitness type
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # register upper and lower bounds
    toolbox.register("attr_float_01", random.uniform, 0, 1)                     # feasible bounds for GED weights, eta and tau_f
    if extractStrategy_Granulator in ['stratSamplePaths', 'stratSampleCliques']:
        toolbox.register("Q", random.randint, 1, round(500 / numClasses))       # feasible bounds for Q (uniform scaling for stratified approaches)
    else:
        toolbox.register("Q", random.randint, 1, 500)                           # feasible bounds for Q

    if extractStrategy_Granulator in ['stratSamplePaths', 'stratSampleCliques']:                    # explicit lists
        lb, ub = [0, 0, 1, 0, 0, 0, 0, 0, 0], [1, 1, round(500 / numClasses), 1, 1, 1, 1, 1, 1]     # (for clipping)
    else:                                                                                           #
        lb, ub = [0, 0, 1, 0, 0, 0, 0, 0, 0], [1, 1, 500, 1, 1, 1, 1, 1, 1]                         #

    # Structure initializers
    toolbox.register("individual", tools.initCycle, creator.Individual, \
        (toolbox.attr_float_01, toolbox.attr_float_01, toolbox.Q, toolbox.attr_float_01, toolbox.attr_float_01, toolbox.attr_float_01, toolbox.attr_float_01, toolbox.attr_float_01, toolbox.attr_float_01), 1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # register the goal / fitness function
    # toolbox.register("evaluate", fitnessfunction_GA1, bucket=bucket, trSet_EXP=trSet_EXP, vsSet_EXP=vsSet_EXP, extractStrategy_Granulator=extractStrategy_Granulator, numClasses=numClasses, nodeDissimilarity=nodeDissimilarity, edgeDissimilarity=edgeDissimilarity)

    # register the crossover operator
    toolbox.register("mate", tools.cxTwoPoint)

    # register the mutation operator
    toolbox.register("mutate", tools.mutGaussian, indpb=1 / len(lb), mu=0, sigma=1)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    toolbox.register("select", tools.selRoulette)

    # perform individual evaluation in parallel
    pool = multiprocessing.Pool(processes=n_threads)
    toolbox.register("map", pool.map)

    # CXPB  is the probability with which two individuals are crossed
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.8, 0.3

    # create initial population
    pop = toolbox.population(n=20)

    # create hall of fame for storing the best overall individual
    hof = tools.HallOfFame(1)

    # statistics to keep track of evolution progress
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    return toolbox, pop, CXPB, MUTPB, hof, stats, lb, ub


def setup_GA2(alphabetSize, n_threads):
    """ Parameters for second GA (feature selection). To be used with DEAP.

    Input:
    - alphabetSize: size of the best alphabet to be pruned
    - n_threads: number of threads for parallel individual evaluation
    Output:
    - toolbox: a DEAP toolbox including the genetic algorithm setup
    - pop: the initial population
    - CXPB: crossover probability
    - MUTPB: mutation probability
    - hof: the DEAP hall of fame for saving the global best individual
    - stats: DEAP module for capturing basic statistics as the evolution goes by
    - lb: chromosomes lower bounds
    - ub: chromosomes upper bounds. """

    # register individual type and fitness type
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # register upper and lower bounds
    toolbox.register("attr_bool", random.randint, 0, 1)                     # feasible bounds for feature selection vector
    lb, ub = [0] * alphabetSize, [1] * alphabetSize                         # explicit lists (for clipping)

    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, alphabetSize)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # register the goal / fitness function
    # toolbox.register("evaluate", fitnessfunction_GA2, trSet_EMB_InstanceMatrix=trSet_EMB_InstanceMatrix, vsSet_EMB_InstanceMatrix=vsSet_EMB_InstanceMatrix, trSet_EMB_LabelVector=trSet_EMB_LabelVector, vsSet_EMB_LabelVector=vsSet_EMB_LabelVector)

    # register the crossover operator
    toolbox.register("mate", tools.cxTwoPoint)

    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutFlipBit, indpb=1 / alphabetSize)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    toolbox.register("select", tools.selTournament, tournsize=3)

    # perform individual evaluation in parallel
    pool = multiprocessing.Pool(processes=n_threads)
    toolbox.register("map", pool.map)

    # CXPB  is the probability with which two individuals are crossed
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.8, 0.3

    # create initial population (100 individuals)
    M = numpy.random.choice(a=[True, False], size=(10, alphabetSize), p=[0.1, 1 - 0.1])                         # 10 individuals with 10% of 1's
    M = numpy.vstack((M, numpy.random.choice(a=[True, False], size=(10, alphabetSize), p=[0.2, 1 - 0.2])))      # 10 individuals with 20% of 1's
    M = numpy.vstack((M, numpy.random.choice(a=[True, False], size=(10, alphabetSize), p=[0.3, 1 - 0.3])))      # 10 individuals with 30% of 1's
    M = numpy.vstack((M, numpy.random.choice(a=[True, False], size=(10, alphabetSize), p=[0.4, 1 - 0.4])))      # 10 individuals with 40% of 1's
    M = numpy.vstack((M, numpy.random.choice(a=[True, False], size=(10, alphabetSize), p=[0.5, 1 - 0.5])))      # 10 individuals with 50% of 1's
    M = numpy.vstack((M, numpy.random.choice(a=[True, False], size=(10, alphabetSize), p=[0.6, 1 - 0.6])))      # 10 individuals with 60% of 1's
    M = numpy.vstack((M, numpy.random.choice(a=[True, False], size=(10, alphabetSize), p=[0.7, 1 - 0.7])))      # 10 individuals with 70% of 1's
    M = numpy.vstack((M, numpy.random.choice(a=[True, False], size=(10, alphabetSize), p=[0.8, 1 - 0.8])))      # 10 individuals with 80% of 1's
    M = numpy.vstack((M, numpy.random.choice(a=[True, False], size=(10, alphabetSize), p=[0.9, 1 - 0.9])))      # 10 individuals with 90% of 1's
    M = numpy.vstack((M, numpy.random.choice(a=[True, False], size=(5, alphabetSize), p=[0.05, 1 - 0.05])))     # 5 individuals with 5% of 1's
    M = numpy.vstack((M, numpy.random.choice(a=[True, False], size=(5, alphabetSize), p=[0.95, 1 - 0.95])))     # 5 individuals with 95% of 1's
    pop = M.astype(int).tolist()
    pop = [creator.Individual(item) for item in pop]

    # create hall of fame for storing the best overall individual
    hof = tools.HallOfFame(1)

    # statistics to keep track of evolution progress
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    return toolbox, pop, CXPB, MUTPB, hof, stats, lb, ub


def fitnessfunction_GA1_DE(genetic_code, *data):
    """ Fitness function for first GA (alphabet tuning). To be used with SciPys' differential_evolution().

    Input:
    - genetic_code: Individual object provided by DEAP
    - data: tuple of additional arguments, namely
        1. bucket: list of subgraphs (if not stratified) or list-of-lists of class-specific subgraphs (if stratified) to be clustered
        2. trSet_EXP: a dictionary of (expanded) training graphs of the form {id: (list-of-subgraphs, label)}
        3. vsSet_EXP: a dictionary of (expanded) validation graphs of the form {id: (list-of-subgraphs, label)}
        4. extractStrategy_Granulator: string representing the granulation strategy (to be set in main.py)
        5. numClasses: number of classes for the classification problem (useful only for stratified approaches)
        6. graphDissimilarity: object endowing the dissimilarity
        7. theta_candidates: list of theta candidates for BSAS
        8. epsilon: tolerance value in symbols recognition
        9. dataName: the name of the dataset (for parsing additional weights, if applicable).
    Output:
    - fitness: fitness value (to be minimised) of the form [alpha * error_rate + (1 - alpha) * number_of_symbols]. """

    # Strip input data
    #bucket, trSet_EXP, vsSet_EXP, extractStrategy_Granulator, numClasses, graphDissimilarity, theta_candidates, epsilon, dataName = data
    label, bucket, trSet_EXP, vsSet_EXP, extractStrategy_Granulator, graphDissimilarity, theta_candidates, epsilon, dataName = data
    #Class relabeling
    #Target class is 1
    # trSet_EXPrelabel={k: (v[0],1) if(v[1]==label) else (v[0],0) for k, v in trSet_EXP.items()}
    # vsSet_EXPrelabel={k: (v[0],1) if(v[1]==label) else (v[0],0) for k, v in vsSet_EXP.items()}
    
    #Strip parameters from genetic code
    eta = genetic_code[0]
    tau_f = genetic_code[1]
    Q = round(genetic_code[2])
    pVertexSubs = genetic_code[3]
    pVertexIns = genetic_code[4]
    pVertexDel = genetic_code[5]
    pEdgeSubs = genetic_code[6]
    pEdgeIns = genetic_code[7]
    pEdgeDel = genetic_code[8]
    

    # nodes/edges dissimilarities setup
    if dataName == 'GREC':
        vParam1 = genetic_code[9]
        eParam1 = genetic_code[10]
        eParam2 = genetic_code[11]
        eParam3 = genetic_code[12]
        eParam4 = genetic_code[13]
        graphDissimilarity._vParam1 = vParam1
        graphDissimilarity._eParam1 = eParam1
        graphDissimilarity._eParam2 = eParam2
        graphDissimilarity._eParam3 = eParam3
        graphDissimilarity._eParam4 = eParam4
    if dataName == 'PROTEIN':
        vParam1 = genetic_code[9]
        eParam1 = genetic_code[10]
        eParam2 = genetic_code[11]
        eParam3 = genetic_code[12]
        eParam4 = genetic_code[13]
        eParam5 = genetic_code[14]
        graphDissimilarity._vParam1 = vParam1
        graphDissimilarity._eParam1 = eParam1
        graphDissimilarity._eParam2 = eParam2
        graphDissimilarity._eParam3 = eParam3
        graphDissimilarity._eParam4 = eParam4
        graphDissimilarity._eParam5 = eParam5   
        
    # GED setup
    Diss = GED(graphDissimilarity.nodeDissimilarity, graphDissimilarity.edgeDissimilarity)
    Diss.nodesParam['sub'] = pVertexSubs
    Diss.nodesParam['ins'] = pVertexIns
    Diss.nodesParam['del'] = pVertexDel
    Diss.edgesParam['sub'] = pEdgeSubs
    Diss.edgesParam['ins'] = pEdgeIns
    Diss.edgesParam['del'] = pEdgeDel
    

    # Set useful parameters
    alpha = 0.9

    # Granulator
    if extractStrategy_Granulator in ['samplePaths', 'paths', 'cliques']:
        ALPHABET = ensembleGranulator(bucket, Diss.BMF, Q, eta, tau_f, theta_candidates, epsilon)
        bucketSize = len(bucket)
    elif extractStrategy_Granulator in ['stratSamplePaths', 'stratSampleCliques']:
        ALPHABET = ensembleStratifiedGranulator(bucket, Diss.BMF, Q, eta, tau_f, numClasses, theta_candidates, epsilon)
        bucketSize = sum([len(bucket[c]) for c in range(numClasses)])

    # Prior exit if alphabet is empty 
    if ALPHABET == []:
        print("Parameters: " + str([round(i, 2) for i in genetic_code]) + "\tEmpty Alphabet")
        return 2

    ALPHABET, tau_k = zip(*ALPHABET)

    # Embedder
    trSet_EMB_InstanceMatrix, trSet_EMB_LabelVector = symbolicHistogramsEmbedder(trSet_EXP, ALPHABET, tau_k, Diss.BMF)
    vsSet_EMB_InstanceMatrix, vsSet_EMB_LabelVector = symbolicHistogramsEmbedder(vsSet_EXP, ALPHABET, tau_k, Diss.BMF)

    # Class relabelling where target class is 1
    trSet_EMB_LabelVector = (trSet_EMB_LabelVector == label).astype(int)
    vsSet_EMB_LabelVector = (vsSet_EMB_LabelVector == label).astype(int)

    # Classifier
    KNN = KNeighborsClassifier(n_neighbors=5)
    KNN.fit(trSet_EMB_InstanceMatrix, trSet_EMB_LabelVector)
    predicted_vsSet = KNN.predict(vsSet_EMB_InstanceMatrix)
    
    #Move to informedness
    """ From sci-kit lib confusion matrix return C as follow:
    [...]Thus in binary classification, the count of true negatives is
    :math:`C_{0,0}`, false negatives is :math:`C_{1,0}`, true positives is
    :math:`C_{1,1}` and false positives is :math:`C_{0,1}`.
    """    
    tn, fp, fn, tp = confusion_matrix(vsSet_EMB_LabelVector, predicted_vsSet).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    J = sensitivity + specificity - 1
    J = (J + 1) / 2
    error_rate = 1 - J

    # accuracy = accuracy_score(vsSet_EMB_LabelVector, predicted_vsSet)
    # error_rate = 1 - accuracy
        
    fitness = alpha * error_rate + (1 - alpha) * (len(ALPHABET) / bucketSize)   # add a small term in order to prefer small alphabets upon (pretty much) the same accuracy
    # print("Parameters: " + str([round(i, 2) for i in genetic_code]) + "\tAccuracy " + str(accuracy) + "\tAlphabet: " + str(len(ALPHABET)))
    print("Parameters: " + str([round(i, 2) for i in genetic_code]) + "\tInformedness " + str(J) + "\tAlphabet: " + str(len(ALPHABET)))
    return fitness


def fitnessfunction_GA2_DE(genetic_code, *data):
    """ Fitness function for second GA (feature selection). To be used with SciPys' differential_evolution().

    Input:
    - genetic_code: Individual object provided by DEAP
    - data: tuple of additional arguments, namely
        1. trSet_EMB_InstanceMatrix: embedded training set with the best alphabet
        2. vsSet_EMB_InstanceMatrix: embedded validation set with the best alphabet
        3. trSet_EMB_LabelVector: training set labels
        4. vsSet_EMB_LabelVector: validation set labels.
    Output:
    - fitness: fitness value (to be minimised) of the form [alpha * error_rate + (1 - alpha) * number_of_selected_symbols]. """

    # Strip input data
    trSet_EMB_InstanceMatrix, vsSet_EMB_InstanceMatrix, trSet_EMB_LabelVector, vsSet_EMB_LabelVector = data

    # Strip parameters from genetic code
    mask = [round(i) for i in genetic_code]

    # Set useful parameters
    alpha = 0.9

    # Evalaute mask cost
    mask = numpy.array(mask, dtype=bool)
    selectedRatio = sum(mask) / len(genetic_code)

    # Prior exit if no features have been selected
    if selectedRatio == 0:
        return 2

    # Classifier
    KNN = KNeighborsClassifier(n_neighbors=5)
    #KNN.fit(trSet_EMB_InstanceMatrix[:, mask], trSet_EMB_LabelVector)

    #Moving to ensemble structure on validation phase
    KNNensemble=OneVsRestClassifier(KNN).fit(trSet_EMB_InstanceMatrix[:, mask], trSet_EMB_LabelVector)
    predicted_vsSet=KNNensemble.predict(vsSet_EMB_InstanceMatrix[:, mask])
    accuracy = accuracy_score(vsSet_EMB_LabelVector, predicted_vsSet)
    ##
    
    # predicted_vsSet = KNN.predict(vsSet_EMB_InstanceMatrix[:, mask])
    # accuracy = accuracy_score(vsSet_EMB_LabelVector, predicted_vsSet)
    error_rate = 1 - accuracy
    fitness = alpha * error_rate + (1 - alpha) * selectedRatio
    print("Selection ratio: " + str(selectedRatio) + "\tAccuracy: " + str(accuracy))
    return fitness


def setup_GA1_DE(n_threads, extractStrategy_Granulator, numClasses, dataName):
    """ Parameters for first GA (alphabet tuning). To be used with SciPy's differential_evolution().

    Input:
    - n_threads: number of threads for parallel individual evaluation
    - extractStrategy_Granulator: string representing the granulation strategy (to be set in main.py)
    - numClasses: number of classes for the classification problem (useful only for stratified approaches)
    - numClasses: the dataset name (for adding additional weights, if applicable)
    Output:
    - bounds: list of tuples of the form (min, max) encoding lower and upper bounds for each variable
    - CXPB: crossover probability
    - MUTPB: mutation probability. """

    # Declare bounds
    bounds = [(0, 1)]                                                       # this is for eta

    bounds = bounds + [(0, 1)]                                              # this is for tau_f

    if extractStrategy_Granulator in ['stratSamplePaths', 'stratSampleCliques']:    # for Q we apply
        bounds = bounds + [(1, round(500 / numClasses))]                            # the uniform scaling
    else:                                                                           # in case of
        bounds = bounds + [(1, 500)]                                                # stratified approaches

    bounds = bounds + list(zip([0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]))     # these are the 6 GED weights

    if dataName == 'GREC':
        bounds = bounds + list(zip([0, 0, 0, 0, 0], [1, 1, 1, 1, 1]))               # these are the 5 weights for GREC
    if dataName == 'PROTEIN':
        bounds = bounds + list(zip([0, 0, 0, 0, 0, 0], [1, 1, 1 ,1 ,1 ,1]))         # these are the 5 weights for PROTEIN

    # CXPB  is the probability with which two individuals are crossed
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.8, 0.3

    return bounds, CXPB, MUTPB


def setup_GA2_DE(alphabetSize, n_threads):
    """ Parameters for second GA (feature selection). To be used with SciPy's differential_evolution().

    Input:
    - alphabetSize: size of the best alphabet to be pruned
    - n_threads: number of threads for parallel individual evaluation
    Output:
    - bounds: list of tuples of the form (min, max) encoding lower and upper bounds for each variable
    - CXPB: crossover probability
    - MUTPB: mutation probability
    - pop: the initial population. """

    # Declare bounds
    bounds = list(zip([0] * alphabetSize, [1] * alphabetSize))

    # CXPB  is the probability with which two individuals are crossed
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.8, 0.3

    # initial population trick (100 individuals)
    M = numpy.random.choice(a=[True, False], size=(10, alphabetSize), p=[0.1, 1 - 0.1])                         # 10 individuals with 10% of 1's
    M = numpy.vstack((M, numpy.random.choice(a=[True, False], size=(10, alphabetSize), p=[0.2, 1 - 0.2])))      # 10 individuals with 20% of 1's
    M = numpy.vstack((M, numpy.random.choice(a=[True, False], size=(10, alphabetSize), p=[0.3, 1 - 0.3])))      # 10 individuals with 30% of 1's
    M = numpy.vstack((M, numpy.random.choice(a=[True, False], size=(10, alphabetSize), p=[0.4, 1 - 0.4])))      # 10 individuals with 40% of 1's
    M = numpy.vstack((M, numpy.random.choice(a=[True, False], size=(10, alphabetSize), p=[0.5, 1 - 0.5])))      # 10 individuals with 50% of 1's
    M = numpy.vstack((M, numpy.random.choice(a=[True, False], size=(10, alphabetSize), p=[0.6, 1 - 0.6])))      # 10 individuals with 60% of 1's
    M = numpy.vstack((M, numpy.random.choice(a=[True, False], size=(10, alphabetSize), p=[0.7, 1 - 0.7])))      # 10 individuals with 70% of 1's
    M = numpy.vstack((M, numpy.random.choice(a=[True, False], size=(10, alphabetSize), p=[0.8, 1 - 0.8])))      # 10 individuals with 80% of 1's
    M = numpy.vstack((M, numpy.random.choice(a=[True, False], size=(10, alphabetSize), p=[0.9, 1 - 0.9])))      # 10 individuals with 90% of 1's
    M = numpy.vstack((M, numpy.random.choice(a=[True, False], size=(5, alphabetSize), p=[0.05, 1 - 0.05])))     # 5 individuals with 5% of 1's
    M = numpy.vstack((M, numpy.random.choice(a=[True, False], size=(5, alphabetSize), p=[0.95, 1 - 0.95])))     # 5 individuals with 95% of 1's
    pop = M.astype(int)

    return bounds, CXPB, MUTPB, pop
