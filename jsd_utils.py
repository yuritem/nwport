import warnings
import numpy as np
import networkx as nx

from portrait_utils import network_portrait, to_same_shape


def c_lk(port):
    return port.copy().cumsum(axis=1) / port.shape[1]


def p_k_given_l(port):
    return port.copy() / port.shape[1]


def ks_stat(C1, C2):
    assert len(C1) == len(C2), "Diameters of cumulative distribution arrays are not equal."
    c_delta = np.abs(np.subtract(C1, C2))
    n_rows = len(c_delta)
    k = np.zeros(n_rows)
    for l_ in range(n_rows):
        k[l_] = max(c_delta[l_])
    return k


def p_kl(G, port=None):
    if port is None:
        warnings.warn("No portrait supplied to p_kl. JSD may be calcualted incorrectly.", RuntimeWarning)
        port = network_portrait(G)
    _, n = port.shape

    if isinstance(G, nx.DiGraph):
        conn_components_sizes = np.asarray([len(c) for c in nx.strongly_connected_components(G)]) ** 2
    else:
        # assumes that isinstance(G, nx.Graph)
        conn_components_sizes = np.asarray([len(c) for c in nx.connected_components(G)]) ** 2
    norm = conn_components_sizes.sum()
    nums = np.arange(0, n)

    port_copy = port.copy()
    p_l = np.sum(np.where(port_copy != 0, port_copy * nums, 0), axis=1) / norm
    p_k_given_l_ = port.copy() / n
    return np.multiply(p_k_given_l_, p_l[:, np.newaxis])


def kld(p, q):
    # Calculates KL-divergence
    # implicitly assumes there are no points where q == 0 and p != 0
    return np.sum(np.where(p != 0, p * np.log2(p / q), 0))


def pdf_mixture(pdf_1, pdf_2):
    return pdf_1 / 2 + pdf_2 / 2


def jsd_distrib(p, q):
    # Calculates JS (Jensen-Shannon) Divergence
    # args: = distributions
    m = pdf_mixture(p, q)
    return kld(p, m) / 2 + kld(q, m) / 2


def jsd_graphs(G_1, G_2):
    # Calculates JS (Jensen-Shannon) Divergence
    # args: graphs with portraits of equal shape
    port_1 = network_portrait(G_1)
    port_2 = network_portrait(G_2)
    port_1, port_2 = to_same_shape([port_1, port_2], zeros=False)
    p = p_kl(G_1, port_1)
    q = p_kl(G_2, port_2)
    m = pdf_mixture(p, q)
    return kld(p, m) / 2 + kld(q, m) / 2


def jsd(*args):
    # syntactic sugar for two JSD versions above
    assert len(args) == 2, f"Too many args passed to jsd()."
    if all(isinstance(arg, nx.Graph) for arg in args) or \
       all(isinstance(arg, nx.DiGraph) for arg in args):
        return jsd_graphs(*args)
    elif all(isinstance(arg, np.ndarray) for arg in args):
        return jsd_distrib(*args)
    else:
        raise TypeError("jsd() got arguments of wrong type."
                        "Please supply graphs or numpy arrays")
