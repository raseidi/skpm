import numpy as np
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    check_is_fitted,
)
from skpm.utils.validation import validate_methods_from_class
from skpm.config import EventLogConfig as elc


class DigraphFeaturesExtractor(TransformerMixin, BaseEstimator):
    """Extracts features from a directed graph represented by event data.

    This class calculates various features from a directed graph extracted from event data.

    Attributes:
    -----------
    frequency_matrix : ndarray
        Transition frequency matrix representing the directed graph.
    stoi : dict
        Mapping of states to indices.
    itos : dict
        Mapping of indices to states.
    features_ : list of tuples
        List of feature names and corresponding feature functions.

    Methods:
    --------
    fit(X, y=None):
        Fit the feature extractor to the input data.
    transform(X, y=None):
        Transform the input data to calculate graph features.

    Notes:
    ------
    This class requires event data with columns 'caseid' and 'activity' for fitting and transformation.

    Examples:
    ---------
    >>> from skpm.event_feature_extraction.meta import DigraphFeaturesExtractor
    >>> from skpm.config import EventLogConfig as elc
    >>> import pandas as pd
    >>> # Assuming X is your dataframe containing event data with columns 'caseid' and 'activity'
    >>> X = pd.DataFrame({elc.case_id: [1, 1, 2, 2], elc.timestamp: ['A', 'B', 'A', 'C']})
    >>> feature_extractor = DigraphFeaturesExtractor()
    >>> feature_extractor.fit_transform(X)
    """

    def __init__(self, features: str = "all") -> None:
        self.features = features

    def fit(self, X, y=None):
        """Fit the feature extractor to the input data.

        Parameters:
        -----------
        X : pd.DataFrame
            Input DataFrame containing event data with columns 'caseid' and 'activity'.
        y : None
            Ignored.

        Returns:
        --------
        self : DigraphFeaturesExtractor
            Returns the instance itself.
        """
        traces = X.groupby(elc.case_id)[elc.activity].apply(list)
        states = set(X[elc.activity].unique())
        (
            self.frequency_matrix,
            self.stoi,
            self.itos,
            # TODO: the bottom line has key error on elc.activity.
        ) = _DigraphFeatures._frequency_matrix(traces=traces, set_of_states=states)

        # TODO: refactor this part; this is hard to debug
        self.features_ = validate_methods_from_class(self.features, _DigraphFeatures)
        self.features_ = set(self.features_) - {"_frequency_matrix"}
        self._n_features_out = len(self.features_)
        return self

    def transform(self, X, y=None):
        """
        Transform the input data to calculate graph features.

        Parameters:
        -----------
        X : pd.DataFrame
            Input DataFrame containing event data with columns 'caseid' and 'activity'.
        y : None
            Ignored.

        Returns:
        --------
        pd.DataFrame
            Transformed DataFrame with calculated graph features added.
        """
        check_is_fitted(self, "features_")

        # TODO: this does not work as a transformer; refactor
        # for feature_name, feature_fn in self.features_:
        #     X[feature_name] = feature_fn(self.frequency_matrix)

        # temporary solution: frequency matrix as dataframe
        import pandas as pd
        states = self.itos.values()
        tmp = pd.DataFrame(self.frequency_matrix, columns=[f"to_{state}" for state in states], index=[f"from_{state}" for state in states])

        return tmp


class _DigraphFeatures:
    @classmethod
    def _frequency_matrix(
            cls, traces: list, set_of_states: set
    ) -> tuple[np.ndarray, dict, dict]:
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
        # TODO: this implementation is permutation-invariant; we should assert order for reproducibility
        stoi = {value: ix for ix, value in enumerate(set_of_states)}
        itos = {ix: value for value, ix in stoi.items()}
        freq_matrix = np.zeros((len(stoi), len(stoi)), dtype=np.int32)

        for transition in traces:
            for origin, destiny in zip(transition, transition[1:]):
                freq_matrix[stoi[origin], stoi[destiny]] += 1

        return freq_matrix, stoi, itos

    @classmethod
    def node_degree(cls, frequency_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

    @classmethod
    def density(cls, graph):
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

    @classmethod
    def nodes_in_cycles(cls, frequency_matrix, max_cycle_length):
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
