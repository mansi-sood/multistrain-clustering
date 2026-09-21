import collections
import random
import sys


import numpy as np
import networkx as nx
from numpy import linalg as LA
from scipy.optimize import fsolve

import sim_core as sc


T1, T2 = 0.2, 0.5
m11 = m22 = 0.75
u11, u12, u22, u21 = m11, 1 - m11, m22, 1 - m22

C11, C12 = (1 - T1) ** 2, (1 - T2) ** 2
C21 = 2 * T1 * u11 * (1 - T1) * (1 - T1)
C22 = 2 * T2 * u21 * (1 - T2) * (1 - T1)
C31 = (T1 * u11) ** 2 + 2 * T1 * u11 * (1 - T1) * T1 * u11
C32 = (T2 * u21) ** 2 + 2 * T2 * u21 * (1 - T2) * T1 * u11
C41 = 2 * T1 * u12 * (1 - T1) * (1 - T2)
C42 = 2 * T2 * u22 * (1 - T2) * (1 - T2)
C51 = (T1 * u12) ** 2 + 2 * T1 * u12 * (1 - T1) * T2 * u22
C52 = (T2 * u22) ** 2 + 2 * T2 * u22 * (1 - T2) * T2 * u22
C61 = 2 * (T1 * u11 * T1 * u12 + T1 * u11 * (1 - T1) * T1 * u12 + T1 * u12 * (1 - T1) * T2 * u21)
C62 = 2 * (T2 * u21 * T2 * u22 + T2 * u21 * (1 - T2) * T1 * u12 + T2 * u22 * (1 - T2) * T2 * u21)

Pi = np.array([[T1 * u11, T1 * u12], [T2 * u21, T2 * u22]])
T_BP = float(max(abs(LA.eigvals(Pi))))


def thin_to_mean_degree(G, target_lambda, rng):
    target_edges = int(round(target_lambda * G.number_of_nodes() / 2))
    edges = list(G.edges())
    if target_edges >= len(edges):
        return G.copy()
    kept = rng.sample(edges, target_edges)
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from(kept)
    return H


def empirical_st_distribution(G):
    tri = nx.triangles(G)
    deg = dict(G.degree())
    counts = collections.Counter()
    for node in G.nodes():
        t_i = tri[node]
        s_i = max(deg[node] - 2 * t_i, 0)
        counts[(s_i, t_i)] += 1
    return counts


class EmpiricalDist:
    def __init__(self, counts):
        n = sum(counts.values())
        self.items = [(s, t, c / n) for (s, t), c in counts.items()]
        self.mean_s = sum(s * p for s, t, p in self.items)
        self.mean_t = sum(t * p for s, t, p in self.items)


def getGs_emp(x, y, dist):
    if dist.mean_s == 0:
        return 1.0
    return sum(s * p / dist.mean_s * (x ** (s - 1)) * (y ** t) for s, t, p in dist.items if s >= 1)


def getGt_emp(x, y, dist):
    if dist.mean_t == 0:
        return 1.0
    return sum(t * p / dist.mean_t * (x ** s) * (y ** (t - 1)) for s, t, p in dist.items if t >= 1)


def getGp_emp(x, y, dist):
    return sum(p * (x ** s) * (y ** t) for s, t, p in dist.items)


def equations(p, dist):
    u1, u2, v1, v2 = p
    Gs1, Gs2 = getGs_emp(u1, v1, dist), getGs_emp(u2, v2, dist)
    Gt1, Gt2 = getGt_emp(u1, v1, dist), getGt_emp(u2, v2, dist)
    return (u1 - (1 - T1 + T1 * u11 * Gs1 + T1 * u12 * Gs2),
            u2 - (1 - T2 + T2 * u21 * Gs1 + T2 * u22 * Gs2),
            v1 - (C11 + C21 * Gt1 + C31 * Gt1 * Gt1 + C41 * Gt2 + C51 * Gt2 * Gt2 + C61 * Gt1 * Gt2),
            v2 - (C12 + C22 * Gt1 + C32 * Gt1 * Gt1 + C42 * Gt2 + C52 * Gt2 * Gt2 + C62 * Gt1 * Gt2))


def cascade_prob_empirical(dist):
    sol = fsolve(equations, (0.01, 0.01, 0.01, 0.01), args=(dist,), xtol=1e-8)
    u1, u2, v1, v2 = sol
    return 1 - getGp_emp(u1, v1, dist)


def epidemic_size_empirical(T_tilde, dist, max_iter=10000, tol=1e-13):


    C1 = (1 - T_tilde) ** 2
    C2 = 2 * T_tilde * (1 - T_tilde) ** 2
    C3 = T_tilde ** 2 * (3 - 2 * T_tilde)
    u, v2 = 0.0, 0.0
    for _ in range(max_iter):
        Gs = getGs_emp(u, v2, dist)
        Gt = getGt_emp(u, v2, dist)
        u_new = 1 - T_tilde + T_tilde * Gs
        v2_new = C1 + C2 * Gt + C3 * Gt ** 2
        if abs(u_new - u) < tol and abs(v2_new - v2) < tol:
            u, v2 = u_new, v2_new
            break
        u, v2 = u_new, v2_new
    return 1 - getGp_emp(u, v2, dist)


def run_one_experiment(G_full, target_lambda, seed):
    rng = random.Random(seed)
    G_thin = thin_to_mean_degree(G_full, target_lambda, rng)
    total, _, _ = sc.evolve_disease(G_thin, [T1, T2], [m11, m22], start_strain=1)
    return total / G_thin.number_of_nodes()


def load_twitch_ptbr(path):
    G = nx.read_edgelist(path, delimiter=',', nodetype=int, data=False, comments='from')
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


if __name__ == '__main__':
    import argparse
    import multiprocessing
    import time
    from joblib import Parallel, delayed

    parser = argparse.ArgumentParser()
    parser.add_argument('-lam', type=float, nargs='+', default=[1, 2, 3, 4, 5, 6])
    parser.add_argument('-e', type=int, default=10000, help='experiments per lambda')
    parser.add_argument('-thrVal', type=float, default=0.05)
    parser.add_argument('-numCores', type=int, default=11)
    parser.add_argument('-out', default='twitch_ptbr_size_prob.txt')
    parser.add_argument('-fineOut', default='twitch_ptbr_size_prob_fine.txt')
    parser.add_argument('-fineN', type=int, default=40,
                         help='number of theory-only points densely sampled between min(lam) and max(lam), '
                              'for smooth pred. lines (no simulation needed at these points)')
    args = parser.parse_args()
    num_cores = min(args.numCores, multiprocessing.cpu_count())

    print(f'T_BP = rho(Pi) = {T_BP:.4f}')
    G_full = load_twitch_ptbr('data/twitch-ptbr/musae_PTBR_edges.csv')
    print(f'Twitch PTBR: N={G_full.number_of_nodes()}, E={G_full.number_of_edges()}')

    rows = []
    for lam in args.lam:
        t0 = time.time()
        rng = random.Random(int(lam * 1000))
        G_thin = thin_to_mean_degree(G_full, lam, rng)
        dist = empirical_st_distribution(G_thin)
        edist = EmpiricalDist(dist)
        pe_theory = cascade_prob_empirical(edist)
        s_theory = epidemic_size_empirical(T_BP, edist)

        sizes = Parallel(n_jobs=num_cores)(
            delayed(run_one_experiment)(G_full, lam, seed=1000000 * int(lam * 10) + i)
            for i in range(args.e))
        epidemics = [s for s in sizes if s >= args.thrVal]
        pe_exp = len(epidemics) / len(sizes)
        s_exp = float(np.mean(epidemics)) if epidemics else 0.0

        print(f'lambda={lam:5.2f}  P_E(theory)={pe_theory:.4f}  P_E(exp,e={args.e})={pe_exp:.4f}  '
              f'S(theory)={s_theory:.4f}  S(exp)={s_exp:.4f}  mean_s={edist.mean_s:.3f}  '
              f'mean_t={edist.mean_t:.3f}  time={time.time()-t0:.1f}s')
        rows.append((lam, pe_theory, pe_exp, s_theory, s_exp))

    with open(args.out, 'w') as f:
        f.write('lambda, P_E_theory, P_E_exp, S_theory, S_exp\n')
        for lam, pt, pe, st, se in rows:
            f.write(f'{lam}, {pt}, {pe}, {st}, {se}\n')
    print(f'Saved {args.out}')

    fine_lams = np.linspace(min(args.lam), max(args.lam), args.fineN)
    fine_rows = []
    for lam in fine_lams:
        rng = random.Random(int(lam * 1000))
        G_thin = thin_to_mean_degree(G_full, lam, rng)
        dist = empirical_st_distribution(G_thin)
        edist = EmpiricalDist(dist)
        pe_theory = cascade_prob_empirical(edist)
        s_theory = epidemic_size_empirical(T_BP, edist)
        fine_rows.append((lam, pe_theory, s_theory))
    with open(args.fineOut, 'w') as f:
        f.write('lambda, P_E_theory, S_theory\n')
        for lam, pt, st in fine_rows:
            f.write(f'{lam}, {pt}, {st}\n')
    print(f'Saved {args.fineOut}')
