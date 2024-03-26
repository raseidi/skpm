import numpy as np
import pytest
from skpm.utils.graph import frequency_matrix, node_degree, density, nodes_in_cycles


@pytest.fixture
def example_traces():
    return [[1, 2, 3], [1, 2, 3, 4]]


@pytest.fixture
def example_set_of_states():
    return {1, 2, 3, 4}


@pytest.fixture
def example_frequency_matrix():
    return np.array([[0, 2, 0, 0],
                     [0, 0, 2, 0],
                     [0, 0, 0, 1],
                     [0, 0, 0, 0]])


def test_frequency_matrix(example_traces, example_set_of_states, example_frequency_matrix):
    freq_matrix, stoi, itos = frequency_matrix(example_traces, example_set_of_states)
    assert np.array_equal(freq_matrix, example_frequency_matrix)
    assert stoi == {1: 0, 2: 1, 3: 2, 4: 3}
    assert itos == {0: 1, 1: 2, 2: 3, 3: 4}


@pytest.fixture
def example_frequency_matrix_node_degree():
    return np.array([[0, 2, 0, 0],
                     [0, 0, 2, 0],
                     [0, 0, 0, 1],
                     [0, 0, 0, 0]])


def test_node_degree(example_frequency_matrix_node_degree):
    in_degree, out_degree = node_degree(example_frequency_matrix_node_degree)
    assert np.array_equal(in_degree, np.array([0, 2, 2, 1]))
    assert np.array_equal(out_degree, np.array([2, 2, 1, 0]))


def test_density():
    graph = np.array([[0, 2, 0, 0],
                      [0, 0, 2, 0],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0]])
    assert density(graph) == 0.4166666666666667


def test_nodes_in_cycles():
    graph = np.array([[0, 1, 0],
                      [0, 0, 1],
                      [1, 0, 0]])
    assert nodes_in_cycles(graph, max_cycle_length=3) == [True, True, True]
