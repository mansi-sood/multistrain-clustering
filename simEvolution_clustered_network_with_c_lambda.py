import argparse
import collections
import multiprocessing
import random
import sys
import time
from multiprocessing import Manager
# import igraph as ig
import networkx as nx
# import site, os, pdb
import numpy as np
from joblib import Parallel, delayed


# Split a list into approximately equal parts
def chunk_it(seq, num_buckets):
    avg_length = len(seq) / float(num_buckets)
    chunks = []
    last = 0.0

    while last < len(seq):
        chunks.append(seq[int(last):int(last + avg_length)])
        last += avg_length

    return chunks


# Interleave sequences
def inter_leave(seq, num):
    interleaved = []
    counter = 0
    for _ in range(len(seq[0])):
        interleaved.append(np.asarray([seq[j][counter] for j in range(num)]))
        counter += 1
    return interleaved


# Create a random clustered graph based on given node degrees
def create_network(c, lambda_, num_nodes):
    # Ensure the number of single edges is even
    while True:
        single_edges = 2 * np.random.poisson((4 - c) / 2 * lambda_, num_nodes)
        if np.sum(single_edges) % 2 == 0:
            break
    # Ensure the number of triangle edges is a multiple of 3
    while True:
        triangle_edges = np.random.poisson(c / 2 * lambda_, num_nodes)
        if np.sum(triangle_edges) % 3 == 0:
            break

    deg_seq = [(i, j) for i, j in zip(single_edges, triangle_edges)]
    return nx.random_clustered_graph(deg_seq)


# Safe division
def safe_div(x, y):
    return x * 1.0 / y if y != 0 else 0


def run_experiment(i, c, lambda_, num_nodes, transmission_list, mutation_prob):
    network = create_network(c, lambda_, num_nodes)  # network = input a real social network
    network = nx.Graph(network)  # Removes parallel edges and self-loops
    total_size, strain_1_size, strain_2_size = evolve_disease(network, transmission_list, mutation_prob)
    fraction_dict[i] = safe_div(total_size, num_nodes)
    infected_per_st_dict[i] = [safe_div(strain_1_size, num_nodes), safe_div(strain_2_size, num_nodes)]


# Determine the new infections and mutations in the network
def infected_rule(infected_neighbors_dict, transmission_list, susceptible_nodes, num_strains, mutation_prob):
    new_infected_nodes = [set() for _ in range(num_strains)]
    if len(infected_neighbors_dict.keys()) != 0:
        for node, infected_neighbor_strains in infected_neighbors_dict.items():
            random.shuffle(infected_neighbor_strains)
            for strain_type in infected_neighbor_strains:
                # Determine if a node get infected
                if random.random() < transmission_list[strain_type]:
                    susceptible_nodes.remove(node)
                    # Determine if mutation occurs
                    if random.random() < mutation_prob[strain_type]:
                        new_infected_nodes[strain_type].add(node)
                    else:
                        mutated_strain = (strain_type + 1) % num_strains
                        new_infected_nodes[mutated_strain].add(node)
                    break
    return new_infected_nodes


# Determine which strain to start with
def determine_starting_strain(num_nodes):
    random_node = int(np.random.randint(0, num_nodes - 1))
    if start_strain == 1:
        return [{random_node}, set()]
    elif start_strain == 2:
        return [set(), {random_node}]
    else:
        raise ValueError("Invalid starting strain value.")


# Dictates how the disease spreads in the network
def evolve_disease(graph, transmission_list, mutation_prob):
    num_nodes = graph.number_of_nodes()
    node_set = set(graph.nodes())
    # if start_strain == 1:
    #     strain_set_1 = {int(np.random.randint(0, num_nodes - 1))}
    #     strain_set_2 = set()
    # elif start_strain == 2:
    #     strain_set_1 = set()
    #     strain_set_2 = {int(np.random.randint(0, num_nodes - 1))}
    # else:
    #     exit()
    strain_list = determine_starting_strain(num_nodes)
    num_strain = len(strain_list)

    susceptible_nodes = node_set
    for strain_set in strain_list:
        susceptible_nodes = susceptible_nodes.difference(strain_set)
    new_nodes_list = strain_list

    while any(new_nodes_list):
        neighbor_dict = collections.defaultdict(list)

        for strain_type, strain_set in enumerate(new_nodes_list):
            strain_neighbors_list = []
            for node in strain_set:
                strain_neighbors_list += graph.neighbors(node)
            if len(strain_neighbors_list) == 0:
                continue
            for node in strain_neighbors_list:
                if node not in susceptible_nodes:
                    continue
                neighbor_dict[node].append(strain_type)
        new_nodes_list = infected_rule(neighbor_dict, transmission_list, susceptible_nodes, num_strain, mutation_prob)

        strain_list = [strain_list[s_idx].union(s) for s_idx, s in enumerate(new_nodes_list)]
    num_infected = sum([len(s) for s in strain_list])
    num_infected1, num_infected2 = map(len, strain_list)
    return num_infected, num_infected1, num_infected2


def parse_args(args):
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('-m', type=float, nargs='+', default=[1],
                        help='np.linspace(0.001, 7, 50) (default); list of mean degree: you can type 1 3 5')
    parser.add_argument('-c', type=float, default=2.00, help='2.00 (default);c')
    parser.add_argument('-l', type=float, default=0.5, help='0.5 (default); lambda')
    parser.add_argument('-n', type=int, default=200, help='10,000 (default); the number of nodes')
    parser.add_argument('-e', type=int, default=150, help='100 (default); the number of experiments')
    parser.add_argument('-t1', type=float, default=0.2, help='0.5 (default); the transmissibility of strain-1')
    parser.add_argument('-t2', type=float, default=0.5, help='0.5 (default); the transmissibility of strain-2')
    parser.add_argument('-m1', type=float, default=0.75, help='0.5 (default); the mutation probability from 1 to 1')
    parser.add_argument('-m2', type=float, default=0.75, help='0.5 (default); the mutation probability from 2 to 2')
    parser.add_argument('-thrVal', type=float, default=0.05,
                        help='0.001 (default); the threshold to consider a component giant')
    parser.add_argument('-numCores', type=int, default=12, help='number of Cores')
    parser.add_argument('-logName', default='logfile', help='The name of the log file')
    parser.add_argument('-i', type=int, default=1, help='1 (default); starting from type-i node')
    return parser.parse_args(args)


if __name__ == '__main__':
    paras = parse_args(sys.argv[1:])
    mean_degree_list = paras.m
    bucketing = False
    #############
    if bucketing:
        meanDegRange = np.arange(0.1, 10, 0.2)
        numBuckets = 5
        meanDegBuckets = chunk_it(meanDegRange, numBuckets)
        meanDegBuckets = inter_leave(meanDegBuckets, numBuckets)
        bucketIdx = 0
        mean_degree_list = meanDegBuckets[bucketIdx]
    ############

    t1 = paras.t1
    t2 = paras.t2
    m1 = paras.m1
    m2 = paras.m2
    num_nodes = paras.n
    numExp = paras.e
    start_strain = paras.i
    c = paras.c
    lambda_ = paras.l
    num_cores = min(paras.numCores, multiprocessing.cpu_count())
    thrVal = paras.thrVal

    T_list = [t1, t2]
    mutation_probability = [m1, m2]
    ff = open(paras.logName + 'Det', 'w+')
    f = open(paras.logName, 'w+')


    a = time.time()
    ttlEpidemicsSize = 0
    numEpidemics = 0
    Epidemics = []
    EpidemicsPerSt = [0, 0, 0]
    fraction_dict = Manager().dict()
    infected_per_st_dict = Manager().dict()
    ttlFrac = 0

    Parallel(n_jobs=num_cores)(
        delayed(run_experiment)(i, c, lambda_, num_nodes, T_list, mutation_probability)
        for i in range(numExp))

    for ii in range(numExp):
        # print('exp', ii)
        # print('intermediate results')
        resultsFrac = ('c: {0} lambda: {1} Size: {2} infSt1: {3} infSt2: {4}\n'
                       .format(c, lambda_, fraction_dict[ii],
                               infected_per_st_dict[ii][0], infected_per_st_dict[ii][1]))

        if fraction_dict[ii] >= thrVal:
            ff.write(resultsFrac)
            ff.flush()
            numEpidemics += 1
            ttlEpidemicsSize += fraction_dict[ii]
            Epidemics.append(fraction_dict[ii])
            EpidemicsPerSt[0] += infected_per_st_dict[ii][0]
            EpidemicsPerSt[1] += infected_per_st_dict[ii][1]

        ttlFrac += fraction_dict[ii]

    if len(Epidemics) == 0:
        Epidemics.append(0)

    print('printing for c = {0} lambda = {1}'.format(c, lambda_))
    results = 'numExp: {0} Threshold: {1} n: {2} c: {3} lambda: {4} Prob: {5} AvgValidSize: {6} StdValidSize: {7} \
    infSt1: {8} infSt2: {9} AvgSize: {10} T: {11} Mu: {12} Time: {13} numEpidemics: {14} \n' \
        .format(numExp, thrVal, num_nodes, c, lambda_, numEpidemics * 1.0 / numExp,
                safe_div(ttlEpidemicsSize * 1.0, numEpidemics), np.std(Epidemics),
                safe_div(EpidemicsPerSt[0], numEpidemics),
                safe_div(EpidemicsPerSt[1], numEpidemics), safe_div(ttlFrac, numEpidemics),
                ' '.join(map(str, T_list)), ' '.join(map(str, mutation_probability)), time.time() - a, numEpidemics)

    print(results)
    f.write(results)
    f.flush()
