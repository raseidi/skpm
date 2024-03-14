import numpy as np

__all__ = ["frequency_matrix", "node_degree"]


def frequency_matrix(traces: list, set_of_states: set) -> tuple[np.ndarray, dict, dict]:
    """
    Returns a transition frequency matrix.

    This function takes a list of traces, where each trace
    is an ordered sequence of states, and computes a transition
    frequency matrix.

    States can be any hashable object, but they must be comparable.
    For instance, a state can be a string, an integer, or a tuple.

    Parameters
    ----------
    traces : list of list of states
        A list of traces, where each trace is a list of states.
    set_of_states : set of states
        A set of all possible states.

    Returns
    -------
    freq_matrix : numpy.ndarray
        A transition frequency matrix.

    stoi : dict
        A dictionary mapping states to indices.

    itos : dict
        A dictionary mapping indices to states.

    Examples
    --------
    >>> traces = [[1, 2, 3], [1, 2, 3, 4]]
    >>> set_of_states = {1, 2, 3, 4}
    >>> frequency_matrix(traces, set_of_states)
    (array([[0, 2, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0]]),
     {1: 0, 2: 1, 3: 2, 4: 3},
     {0: 1, 1: 2, 2: 3, 3: 4})

    >>> traces = [["a", "b", "c"], ["a", "b", "c", "d"]]
    >>> set_of_states = {"a", "b", "c", "d"}
    >>> frequency_matrix(traces, set_of_states)
    (array([[0, 0, 2, 0],
            [2, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0]]),
     {'b': 0, 'a': 1, 'c': 2, 'd': 3},
     {0: 'b', 1: 'a', 2: 'c', 3: 'd'})

    >>> traces = [[("a", "b"), ("b", "c")], [("a", "b"), ("b", "c"), ("c", "d")]]
    >>> set_of_states = {("a", "b"), ("b", "c"), ("c", "d")}
    >>> frequency_matrix(traces, set_of_states)
    (array([[0, 0, 0],
            [1, 0, 0],
            [0, 2, 0]]),
     {('c', 'd'): 0, ('b', 'c'): 1, ('a', 'b'): 2},
     {0: ('c', 'd'), 1: ('b', 'c'), 2: ('a', 'b')})

    """
    stoi = {value: ix for ix, value in enumerate(set_of_states)}
    itos = {ix: value for value, ix in stoi.items()}
    freq_matrix = np.zeros((len(stoi), len(stoi)), dtype=np.int32)

    for transition in traces:
        for origin, destiny in zip(transition, transition[1:]):
            freq_matrix[stoi[origin], stoi[destiny]] += 1

    return freq_matrix, stoi, itos


def node_degree(frequency_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the in-degree and out-degree of each node.

    Parameters
    ----------
    frequency_matrix : numpy.ndarray
        A graph as a transition frequency matrix.

    Returns
    -------
    in_degree : numpy.ndarray
        An array with the in-degree of each node.

    out_degree : numpy.ndarray
        An array with the out-degree of each node.
    """
    in_degree = frequency_matrix.sum(axis=0)
    out_degree = frequency_matrix.sum(axis=1)

    return in_degree, out_degree


def density(graph):
    """
    Returns the density of a graph.

    Parameters
    ----------
    graph : numpy.ndarray
        A graph as a transition frequency matrix.

    Returns
    -------
    density : float
        The density of the graph.
    """
    n_nodes = graph.shape[0]
    n_edges = graph.sum(axis=None)
    max_edges = n_nodes * (n_nodes - 1)
    density = n_edges / max_edges
    return density


def nodes_in_cycles(frequency_matrix, max_cycle_length):
    """
    Returns a list of whether each node is in a cycle.

    Notice: this function actually returns self-loops, not cycles.
    By definition, a cycle is a path that starts and ends at the same node
    and visits each node at most once. A self-loop is an edge that connects
    a node to itself. A self-loop is a cycle of length 1.


    Parameters
    ----------
    frequency_matrix : numpy.ndarray
        A graph as a transition frequency matrix.

    max_cycle_length: int
        The maximum length of a cycle to be counted.

    Returns
    -------
    in_cycle : list of bool
        A list of whether each node is in a cycle.

    """
    frequency_matrix = np.array(frequency_matrix)
    num_nodes = frequency_matrix.shape[0]
    in_cycle = [
        False
    ] * num_nodes  # Initialize list to store whether each node is in a cycle

    for n in range(2, max_cycle_length + 1):
        matrix_power = np.linalg.matrix_power(frequency_matrix, n)
        for i in range(num_nodes):
            if matrix_power[i, i] > 0:
                in_cycle[
                    i
                ] = True  # Mark node i as in a cycle if diagonal entry is non-zero

    return in_cycle
