import collections
import numpy as np
import networkx as nx


def network_portrait(G, trim_lengths=True, trim_numbers=False, fill_first_col=False):
    lengths = dict(nx.all_pairs_shortest_path_length(G))

    # B_{ℓ,k} ≡ the number of nodes who have k nodes at distance ℓ
    n_nodes = G.number_of_nodes()
    res = np.zeros((n_nodes, n_nodes), dtype=np.int32)

    for key, data in lengths.items():
        counters = collections.Counter(data.values())
        for dist in counters:
            if dist == 0:
                continue
            res[dist, counters[dist]] += 1

    res[0, 1] = n_nodes

    if trim_lengths:
        res = res[~np.all(res == 0, axis=1)]

    if trim_numbers:
        emptys = np.all(res == 0, axis=0)
        i = 0
        for e in emptys[::-1]:
            if not e:
                break
            i += 1

        res = res[:, :-i]

    if fill_first_col:
        for i in range(len(res)):
            res[i, 0] = n_nodes - res[i].sum()

    return res


def to_same_shape(arrays, zeros=False):
    # zeros argument indicates wether we want to just append zero rows or rows with B_{0,k}=N or p_{0,k}=1.
    # implicitly assumes that all arrays have same width
    max_height = max((arr.shape[0] for arr in arrays))
    width = arrays[0].shape[1]
    val = 0. if zeros else width

    for i in range(len(arrays)):
        n_rows_to_append = max_height - len(arrays[i])
        if n_rows_to_append > 0:
            arr_to_append = np.zeros((n_rows_to_append, width), dtype=np.int32)
            arr_to_append[:, 0] = val
            arrays[i] = np.vstack((arrays[i], arr_to_append))

    return arrays


def portraits_are_equivalent(port_1, port_2):
    if port_1.shape[1] != port_2.shape[1]:
        return False
    port_1 = port_1[~np.all(port_1 == 0, axis=1)]
    port_2 = port_2[~np.all(port_2 == 0, axis=1)]
    if (port_1.shape == port_2.shape) and (port_1 == port_2).all():
        return True
    return False


def all_portraits_are_different(portraits):
    n_portraits = len(portraits)
    for i_start in range(n_portraits - 1):
        if any([
            portraits_are_equivalent(portraits[i_start], portraits[i])
            for i in range(i_start + 1, n_portraits)
        ]):
            return False
    return True


def avg_portrait(portraits):
    return np.sum(portraits, axis=0) / len(portraits)
