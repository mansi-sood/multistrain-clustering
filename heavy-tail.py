import argparse
import collections
import math
import multiprocessing
import random
import sys
import time
from multiprocessing import Manager
import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from mpmath import polylog

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


"""--Power-law-with-cutoff degree sampler (see power-law-initialitation.png)--"""

def _li(s, x):
    return complex(polylog(s, x)).real

def power_law_zeta(alpha, Gamma):
    return _li(alpha, math.exp(-1.0 / Gamma))

def power_law_mean_degree(alpha, Gamma):
    c = math.exp(-1.0 / Gamma)
    zeta = _li(alpha, c)
    return _li(alpha - 1, c) / zeta

def build_power_law_pmf(alpha, Gamma, tail_tol=1e-10, max_k_cap=2_000_000):
    zeta = power_law_zeta(alpha, Gamma)
    maxK = max(64, int(20 * Gamma))
    k = None
    pmf = None
    while True:
        k = np.arange(1, maxK + 1)
        pmf = k.astype(float) ** (-alpha) * np.exp(-k / Gamma) / zeta
        tail_mass = 1.0 - pmf.sum()
        if tail_mass < tail_tol or maxK >= max_k_cap:
            break
        maxK *= 2
    pmf = pmf / pmf.sum()
    return k, pmf

def sample_power_law(k, pmf, size):
    return np.random.choice(k, size=size, p=pmf)


# Create a random clustered graph based on given node degrees
def create_network(k_s, pmf_s, k_t, pmf_t, num_nodes):
    # Ensure the number of single edges is even
    while True:
        single_edges = sample_power_law(k_s, pmf_s, num_nodes)
        if np.sum(single_edges) % 2 == 0:
            break
    # Ensure the number of triangle edges is a multiple of 3
    while True:
        triangle_edges = sample_power_law(k_t, pmf_t, num_nodes)
        if np.sum(triangle_edges) % 3 == 0:
            break

    deg_seq = [(int(i), int(j)) for i, j in zip(single_edges, triangle_edges)]
    return nx.random_clustered_graph(deg_seq)


# Safe division
def safe_div(x, y):
    return x * 1.0 / y if y != 0 else 0


def run_experiment(i, k_s, pmf_s, k_t, pmf_t, num_nodes, transmission_list, mutation_prob):
    network = create_network(k_s, pmf_s, k_t, pmf_t, num_nodes)  # network = input a real social network
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


def default_gamma_grid():
    # identical grid to prob-doubly-power-law-theory.py / doubly-power-law-prob-emergence.png
    return list(np.logspace(math.log10(0.3), math.log10(1e7), 30))


def parse_args(args):
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('-gamma', type=float, nargs='+', default=default_gamma_grid(),
                        help='list of exponential-cutoff values Gamma to sweep (single-edge and triangle layers use '
                             'the same Gamma, mirroring lambda_s=lambda_t); defaults to the same 30-point grid used '
                             'in prob-doubly-power-law-theory.py / doubly-power-law-prob-emergence.png. Large Gamma '
                             '(>~1e3) makes the network increasingly heavy-tailed/hub-dominated and slower to '
                             'simulate -- see the module docstring above.')
    parser.add_argument('-alpha', type=float, default=2.05,
                        help='2.05 (default); shared power-law exponent for both layers (alpha_s = alpha_t)')
    parser.add_argument('-n', type=int, default=200000, help='200000 (default); the number of nodes')
    parser.add_argument('-e', type=int, default=15000, help='15000 (default); the number of experiments')
    parser.add_argument('-t1', type=float, default=0.2, help='0.2 (default); the transmissibility of strain-1')
    parser.add_argument('-t2', type=float, default=0.5, help='0.5 (default); the transmissibility of strain-2')
    parser.add_argument('-m1', type=float, default=0.75, help='0.75 (default); the mutation probability from 1 to 1')
    parser.add_argument('-m2', type=float, default=0.75, help='0.75 (default); the mutation probability from 2 to 2')
    parser.add_argument('-thrVal', type=float, default=0.05,
                        help='0.001 (default); the threshold to consider a component giant')
    parser.add_argument('-numCores', type=int, default=12, help='number of Cores')
    parser.add_argument('-logName', default='power_law_sim_log', help='The name of the log file')
    parser.add_argument('-i', type=int, default=1, help='1 (default); starting from type-i node')
    return parser.parse_args(args)


if __name__ == '__main__':
    paras = parse_args(sys.argv[1:])
    gamma_list = paras.gamma
    alpha = paras.alpha

    t1 = paras.t1
    t2 = paras.t2
    m1 = paras.m1
    m2 = paras.m2
    num_nodes = paras.n
    numExp = paras.e
    start_strain = paras.i
    num_cores = min(paras.numCores, multiprocessing.cpu_count())
    thrVal = paras.thrVal

    T_list = [t1, t2]
    mutation_probability = [m1, m2]
    ff = open(paras.logName + 'Det', 'w+')
    f = open(paras.logName, 'w+')

    for Gamma in gamma_list:
        lambda_theory = power_law_mean_degree(alpha, Gamma)
        k_layer, pmf_layer = build_power_law_pmf(alpha, Gamma)
        a = time.time()
        ttlEpidemicsSize = 0
        numEpidemics = 0
        Epidemics = []
        EpidemicsPerSt = [0, 0, 0]
        fraction_dict = Manager().dict()
        infected_per_st_dict = Manager().dict()
        ttlFrac = 0

        Parallel(n_jobs=num_cores)(
            delayed(run_experiment)(i, k_layer, pmf_layer, k_layer, pmf_layer, num_nodes, T_list, mutation_probability)
            for i in range(numExp))

        for ii in range(numExp):
            resultsFrac = ('alpha: {0} Gamma: {1} lambda_theory: {2} Size: {3} infSt1: {4} infSt2: {5}\n'
                           .format(alpha, Gamma, lambda_theory, fraction_dict[ii],
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

        print('printing for alpha =', alpha, 'Gamma =', Gamma)
        results = 'numExp: {0} Threshold: {1} n: {2} alpha: {3} Gamma: {4} lambda_theory: {5} Prob: {6} \
        AvgValidSize: {7} StdValidSize: {8} infSt1: {9} infSt2: {10} AvgSize: {11} T: {12} Mu: {13} Time: {14} \
        numEpidemics: {15} \n' \
            .format(numExp, thrVal, num_nodes, alpha, Gamma, lambda_theory, numEpidemics * 1.0 / numExp,
                    safe_div(ttlEpidemicsSize * 1.0, numEpidemics), np.std(Epidemics),
                    safe_div(EpidemicsPerSt[0], numEpidemics),
                    safe_div(EpidemicsPerSt[1], numEpidemics), safe_div(ttlFrac, numEpidemics),
                    ' '.join(map(str, T_list)), ' '.join(map(str, mutation_probability)), time.time() - a, numEpidemics)

        print(results)
        f.write(results)
        f.flush()
