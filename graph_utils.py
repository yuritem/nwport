import copy
import random
import numpy as np
import scipy as sp
import networkx as nx


class EdgeSwapGraph(nx.Graph):
    def randomize_by_edge_swaps(self, num_iterations):
        newgraph = self.copy()
        edge_list = newgraph.edges()
        num_edges = len(edge_list)
        total_iterations = num_edges * num_iterations

        for i in range(total_iterations):
            rand_index1 = int(round(random.random() * (num_edges - 1)))
            rand_index2 = int(round(random.random() * (num_edges - 1)))
            original_edge1 = edge_list[rand_index1]
            original_edge2 = edge_list[rand_index2]
            head1, tail1 = original_edge1
            head2, tail2 = original_edge2

            if random.random() >= 0.5:
                head1, tail1 = tail1, head1

            if head1 == tail2 or head2 == tail1:
                continue

            if newgraph.has_edge(head1, tail2) or newgraph.has_edge(
                    head2, tail1):
                continue

            original_edge1_data = newgraph[head1][tail1]
            original_edge2_data = newgraph[head2][tail2]

            newgraph.remove_edges_from((original_edge1, original_edge2))

            new_edge1 = (head1, tail2, original_edge1_data)
            new_edge2 = (head2, tail1, original_edge2_data)
            newgraph.add_edges_from((new_edge1, new_edge2))

            edge_list[rand_index1] = (head1, tail2)
            edge_list[rand_index2] = (head2, tail1)

        assert len(newgraph.edges()) == num_edges
        return newgraph


def largest_strongly_connected_component(G):
    nodes = max(nx.strongly_connected_components(G), key=len)
    return G.subgraph(nodes).copy()


def lscc(G):  # alias
    return largest_strongly_connected_component(G)


def largest_connected_component(G):
    nodes = max(nx.connected_components(G), key=len)
    return G.subgraph(nodes).copy()


def lcc(G):
    return largest_connected_component(G)


def assign_random_weights(array):
    x_arr = np.random.random(size=(array.shape[0], array.shape[0]))
    w_arr = np.multiply(x_arr, array)
    return (w_arr + w_arr.T)/2


def turn_to_directed(mat, directed=0.0, weighted=0):
    if not isinstance(mat, np.ndarray):
        raise Exception('Wrong input parsed to turn_to_directed function!')

    array = copy.deepcopy(mat)
    if directed == 0.0:
        if not weighted:
            a = array.astype(bool)
        else:
            a = array.astype(float)
        return sp.csr_matrix(a)

    np.fill_diagonal(array, 0)
    rows, cols = array.nonzero()
    edgeset = set(zip(rows, cols))
    upper = np.array([edge for edge in edgeset if edge[0] < edge[1]])

    random_tosses = np.random.random(len(upper))
    condition1 = (random_tosses >= directed / 2.0) & (random_tosses < directed)
    condition2 = (random_tosses <= directed / 2.0) & (random_tosses < directed)
    indices_where_upper_is_removed = np.where(condition1)[0]
    indices_where_lower_is_removed = np.where(condition2)[0]

    u_xdata = [u[0] for u in upper[indices_where_upper_is_removed]]
    u_ydata = [u[1] for u in upper[indices_where_upper_is_removed]]
    array[u_xdata, u_ydata] = 0

    l_xdata = [u[1] for u in upper[indices_where_lower_is_removed]]
    l_ydata = [u[0] for u in upper[indices_where_lower_is_removed]]
    array[l_xdata, l_ydata] = 0

    a = sp.csr_matrix(array)
    return a


def get_symmetry_index(array):

    array = array.astype(bool)
    symmetrized = array + array.T

    difference = symmetrized.astype(int) - array.astype(int)
    difference.eliminate_zeros()

    # symm_index is 1 for a symmetrix matrix and 0 for an asymmetric one
    symm_index = 1 - difference.nnz / symmetrized.nnz * 2

    return symm_index


def symmetric_component(array, is_weighted):
    a = array.astype(bool).A
    symm_mask = np.bitwise_and(a, a.T)
    if not is_weighted:
        return symm_mask

    return np.multiply(symm_mask, array.A)


def non_symmetric_component(array, is_weighted):
    return array.astype(float) - symmetric_component(array, is_weighted).astype(float)


def adj_random_rewiring_iom_preserving(a, is_weighted, r=10):

    s = symmetric_component(a, is_weighted)
    rs = turn_to_directed(s, directed=1.0, weighted=is_weighted)
    rows, cols = rs.A.nonzero()
    edgeset = set(zip(rows, cols))
    upper = [edge for edge in edgeset]
    source_nodes = [e[0] for e in upper]
    target_nodes = [e[1] for e in upper]

    double_edges = len(upper)

    i = 0

    while i < double_edges * r:
        good_choice = 0
        n1, n2, n3, n4, ind1, ind2 = [None] * 6
        while not good_choice:
            ind1, ind2 = np.random.choice(double_edges, 2)
            n1, n3 = source_nodes[ind1], source_nodes[ind2]
            n2, n4 = target_nodes[ind1], target_nodes[ind2]

            if len({n1, n2, n3, n4}) == 4:
                good_choice = 1

        w1 = s[n1, n2]
        w2 = s[n2, n1]
        w3 = s[n3, n4]
        w4 = s[n4, n3]

        if s[n1, n3] + s[n1, n4] + s[n2, n3] + s[n2, n4] == 0:
            s[n1, n4] = w1
            s[n4, n1] = w2
            s[n2, n3] = w3
            s[n3, n2] = w4

            s[n1, n2] = 0
            s[n2, n1] = 0
            s[n3, n4] = 0
            s[n4, n3] = 0

            target_nodes[ind1], target_nodes[ind2] = n4, n2
            i += 1

    # plt.matshow(s)
    # print ('Rewiring single connections...')

    ns = non_symmetric_component(a, is_weighted)

    # plt.matshow(ns)
    rows, cols = ns.nonzero()
    edges = list((set(zip(rows, cols))))
    source_nodes = [e[0] for e in edges]
    target_nodes = [e[1] for e in edges]
    single_edges = len(edges)

    i = 0

    while i < single_edges * r:
        good_choice = 0
        n1, n2, n3, n4, ind1, ind2 = [None] * 6
        while not good_choice:
            ind1, ind2 = np.random.choice(single_edges, 2)
            n1, n3 = source_nodes[ind1], source_nodes[ind2]
            n2, n4 = target_nodes[ind1], target_nodes[ind2]

            if len({n1, n2, n3, n4}) == 4:
                good_choice = 1

        w1 = ns[n1, n2]
        w2 = ns[n3, n4]

        checklist = [ns[n1, n3], ns[n1, n4], ns[n2, n3], ns[n2, n4],
                     ns[n3, n1], ns[n4, n1], ns[n3, n2], ns[n4, n2],
                     s[n3, n1], s[n4, n1], s[n3, n2], s[n4, n2]]

        if checklist.count(0) == 12:
            ns[n1, n4] = w1
            ns[n3, n2] = w2

            ns[n1, n2] = 0
            ns[n3, n4] = 0

            i += 1

            target_nodes[ind1], target_nodes[ind2] = n4, n2

    res = s + ns
    if not is_weighted:
        res = res.astype(bool)

    return sp.csr_matrix(res)


def create_sbm(n, q, w_in=100, w_out=0.01, random_sizes=0, weighted=1):

    if not weighted:
        raise Exception('Unweighted sbm is not implemented')

    a = np.zeros((n, n))

    if random_sizes:
        sizes_do_not_fit = 1
        starts = None
        while sizes_do_not_fit:
            starts = [0] + list(np.sort(np.random.randint(0, high=n, size=q)))
            sz = [starts[i+1] - starts[i] for i in range(len(starts)-1)]
            if sum(np.array([s > n//(2*q) for s in sz]).astype(int)) == q:
                sizes_do_not_fit = 0

    else:
        sz = int(n/q)
        starts = [i*sz for i in range(q)]

    ends = np.r_[[starts[i] - 1 for i in range(1, q)], [n-1]]

    for i in range(q):
        for j in range(i, q):
            if i == j:
                lm = w_in
            else:
                lm = w_out

            a[starts[i]:ends[i]+1, starts[j]:ends[j]+1] = np.random.poisson(lm, size=np.shape(a[starts[i]:ends[i]+1, starts[j]:ends[j]+1]))
            a[starts[j]:ends[j]+1, starts[i]:ends[i]+1] = a[starts[i]:ends[i]+1, starts[j]:ends[j]+1].T

            if i == j:
                a[starts[i]:ends[i]+1, starts[j]:ends[j]+1] = (a[starts[i]:ends[i]+1, starts[j]:ends[j]+1]+a[starts[i]:ends[i]+1, starts[j]:ends[j]+1].T)/2

    return a


def get_single_double_edges_lists(g):
    list_single = []
    list_double = []
    h = nx.to_undirected(g).copy()
    for e in h.edges():
        if g.has_edge(e[1], e[0]):
            if g.has_edge(e[0], e[1]):
                list_double.append((e[0], e[1]))
            else:
                list_single.append((e[1], e[0]))
        else:
            list_single.append((e[0], e[1]))

    return [list_single, list_double]


def random_rewiring_iom_preserving_undirected_unweighted(graph, r=10):

    [list_single, list_double] = get_single_double_edges_lists(graph)
    number_of_single_edges = len(list_single)
    number_of_double_edges = len(list_double)
    number_of_rewired_1_edge_pairs = number_of_single_edges * r
    number_of_rewired_2_edge_pairs = number_of_double_edges * r

    print(f"number_of_rewired_1_edge_pairs: {number_of_rewired_1_edge_pairs}")
    print(f"number_of_rewired_2_edge_pairs: {number_of_rewired_2_edge_pairs}")

    i = 0
    previous_text = ""

    print('Rewiring double connections...')
    while i < number_of_rewired_2_edge_pairs:
        edge_index_1 = random.randrange(0, number_of_double_edges)
        edge_index_2 = random.randrange(0, number_of_double_edges)
        edge_1 = list_double[edge_index_1]
        edge_2 = list_double[edge_index_2]
        [node_a, node_b] = edge_1
        [node_c, node_d] = edge_2
        while (node_a == node_c) or (node_a == node_d) or (node_b == node_c) or (node_b == node_d):
            edge_index_1 = random.randrange(0, number_of_double_edges)
            edge_index_2 = random.randrange(0, number_of_double_edges)
            edge_1 = list_double[edge_index_1]
            edge_2 = list_double[edge_index_2]
            [node_a, node_b] = edge_1
            [node_c, node_d] = edge_2

        if graph.has_edge(node_a, node_d) == 0 and graph.has_edge(node_c, node_b) == 0:
            graph.remove_edge(node_a, node_b)
            graph.remove_edge(node_c, node_d)

            graph.add_edge(node_a, node_d)
            graph.add_edge(node_c, node_b)

            list_double[edge_index_1] = (node_a, node_d)
            list_double[edge_index_2] = (node_c, node_b)
            i += 1

        if (i != 0) and (i % (number_of_double_edges // 1)) == 0:
            text = str(round(100.0 * i / number_of_rewired_2_edge_pairs, 0)) + "%"
            if text != previous_text:
                pass
            previous_text = text

    i = 0
    print('Rewiring single connections...')
    while i < number_of_rewired_1_edge_pairs:
        edge_index_1 = random.randrange(0, number_of_single_edges)
        edge_index_2 = random.randrange(0, number_of_single_edges)
        edge_1 = list_single[edge_index_1]
        edge_2 = list_single[edge_index_2]
        [node_a, node_b] = edge_1
        [node_c, node_d] = edge_2
        while (node_a == node_c) or (node_a == node_d) or (node_b == node_c) or (node_b == node_d):
            edge_index_1 = random.randint(0, number_of_single_edges-1)
            edge_index_2 = random.randint(0, number_of_single_edges-1)
            edge_1 = list_single[edge_index_1]
            edge_2 = list_single[edge_index_2]
            [node_a, node_b] = edge_1
            [node_c, node_d] = edge_2

        if graph.has_edge(node_a, node_d) == 0 and graph.has_edge(node_c, node_b) == 0:
            graph.remove_edge(node_a, node_b)
            graph.remove_edge(node_c, node_d)

            graph.add_edge(node_a, node_d)
            graph.add_edge(node_c, node_b)

            list_single[edge_index_1] = (node_a, node_d)
            list_single[edge_index_2] = (node_c, node_b)
            i += 1

    graph_rewired = copy.deepcopy(graph)

    return graph_rewired
