import networkx as nx
from tqdm import tqdm

from config import FLOAT_PRECISION
from portrait_utils import network_portrait, avg_portrait, to_same_shape


def er_portrait(n, p, n_ensemble=None):
    if not((n_ensemble is None) or (n_ensemble <= 1)):
        graphs_ensemble = [nx.erdos_renyi_graph(n, p, seed=None, directed=False) for _ in range(n_ensemble)]
        portraits_ensemble = to_same_shape([network_portrait(g) for g in graphs_ensemble])
        return avg_portrait(portraits_ensemble)
    else:
        return network_portrait(nx.erdos_renyi_graph(n, p, seed=None, directed=False))


def er_portraits(n, p_range, n_ensemble=None, p_precision=None):
    portraits = []
    labels = []
    fmt_str = f".{p_precision}f" if (p_precision is not None) else ""
    for p in p_range:
        portraits.append(er_portrait(n, p, n_ensemble=n_ensemble))
        labels.append(f"ER N={n}, p={round(p, FLOAT_PRECISION):{fmt_str}}")
        print(labels[-1])
    return portraits, labels


def rrg_portrait(n, d, n_ensemble=None):
    if not((n_ensemble is None) or (n_ensemble <= 1)):
        graphs_ensemble = [nx.random_regular_graph(d, n, seed=None) for _ in range(n_ensemble)]
        portraits_ensemble = to_same_shape([network_portrait(g) for g in graphs_ensemble])
        return avg_portrait(portraits_ensemble)
    else:
        return network_portrait(nx.random_regular_graph(d, n, seed=None))


def rrg_portraits(n, d_range, n_ensemble=None):
    portraits = []
    labels = []
    for d in d_range:
        portraits.append(rrg_portrait(n, d, n_ensemble=n_ensemble))
        labels.append(f"RRG N={n}, d={d}")
        print(labels[-1])
    return portraits, labels


def ba_portrait(n, m, n_ensemble=None):
    if not((n_ensemble is None) or (n_ensemble <= 1)):
        graphs_ensemble = [nx.barabasi_albert_graph(n, m, seed=None, initial_graph=None) for _ in range(n_ensemble)]
        portraits_ensemble = to_same_shape([network_portrait(g) for g in graphs_ensemble])
        return avg_portrait(portraits_ensemble)
    else:
        return network_portrait(nx.barabasi_albert_graph(n, m, seed=None, initial_graph=None))


def ba_portraits(n, m_range, n_ensemble):
    portraits = []
    labels = []
    pbar = tqdm(m_range)
    for m in pbar:
        pbar.set_description("Barabasi-Albert Portraits")
        portraits.append(ba_portrait(n, m, n_ensemble=n_ensemble))
        labels.append(f"BA N={n}, m={m}")
        print(labels[-1])
    return portraits, labels


def ws_portrait(n, k, p, n_ensemble=None):
    if not((n_ensemble is None) or (n_ensemble <= 1)):
        graphs_ensemble = [nx.watts_strogatz_graph(n, k, p, seed=None) for _ in range(n_ensemble)]
        portraits_ensemble = to_same_shape([network_portrait(g) for g in graphs_ensemble])
        return avg_portrait(portraits_ensemble)
    else:
        return network_portrait(nx.watts_strogatz_graph(n, k, p, seed=None))


def ws_portraits(n, k, p_range, n_ensemble=None, p_precision=None):
    portraits = []
    labels = []
    fmt_str = f".{p_precision}f" if (p_precision is not None) else ""
    pbar = tqdm(p_range)
    for p in pbar:
        pbar.set_description("Watts-Strogatz Portraits")
        portraits.append(ws_portrait(n, k, p, n_ensemble=n_ensemble))
        labels.append(f"WS N={n}, k={k}, p={round(p, FLOAT_PRECISION):{fmt_str}}")
    return portraits, labels
