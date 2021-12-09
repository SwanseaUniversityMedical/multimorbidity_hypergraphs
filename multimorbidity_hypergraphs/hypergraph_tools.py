"""
This is a module of tools for use in analysing 2D tables of boolean data
(my use case is multimorbidity, so rows represent people and columns represent
the diseases they do or do not have) by constructing and analysing hypergraph
data structures.
"""

from itertools import combinations, chain

# numpy imports
# submodules
from numpy import random
from numpy.linalg import norm
# types
from numpy import int8, int32, uint8, int64, float64
# array tools
from numpy import zeros, zeros_like, ones_like, array, arange
# functions
from numpy import ceil, log, sqrt, multiply, abs, unique, where

import numba # imported types would clash with some of the numpy imports.

from scipy.sparse import lil_matrix, csr_matrix, diags
from scipy.sparse.linalg import eigsh

from time import time


##########################################
## Numba compiled functions here

@numba.jit(
    nopython=True,
    nogil=True,
    fastmath=True,
)
def _binomial_rvs(N, p):
    """
    Generates a single random rate from a binomial distribution given
    a population size and a probability.
    """
    N_orig = N
    count = 0
    while True:
        wait = ceil(log(random.rand()) / log(1-p))

        if wait > N:
            return count / N_orig
        count += 1
        N -= wait


@numba.jit(
    nopython=True,
    nogil=True,
    parallel=True,
    fastmath=True,
)
def randomize_weights(N_array, p_array, randomisation_fn=_binomial_rvs):
    """
    Apply a random perturbation to a set of weights.
    
    Given an array of populations and probabilities, this function returns 
    the equivalent of:

    [sst.<distribution>(N, p).rvs(samples) / N for N, p in zip(N_array, p_array)]

    <distribution> is binomial by default.

    Parameters
    ----------

        N_array : numpy.array
            An array of population sizes

        p_array : numpy.array
            An array of probabilities

        randomization_fn : callable, jit compiled function, optional
            A function that takes the population and a probaility and returns a 
            randomly perturbed weight (input arguments subject to change, since
            this is currently binomial by default).

    Returns
    -------

        numpy.array : an array of random numbers of length len(N_array)
    """
    out = zeros(len(N_array), dtype=float64)
    for i in numba.prange(len(N_array)):
        out[i] = randomisation_fn(N_array[i], p_array[i])
    return out


@numba.jit(
    nopython=True,
    nogil=True,
    fastmath=True,
)
def _wilson_interval(num, denom):
    """
    Returns an estimate of the variance of the overlap coefficient (and other
    quantities that feature the some number of people divided by the total population
    size) derived from the Wilson score interval.

    Parameters
    ----------

        num : numeric
            the number of "successes", i.e. the numerator of the rate formula we want
            to calculate the variance for.

        denom  : numeric
            the population size of the rate formula we want to calculate the
            variance for.

    Returns
    -------

        numpy.float64 : the estimate of the variance.
    """

    z = 1.959963984540054 # this is the 2.5th/97.5th percentile of the standard normal
    wilson_centre = (num + 0.5 * z ** 2) / (denom + z ** 2)
    wilson_offset = z * sqrt(num * (denom - num) / denom + z ** 2 / 4) / (denom + z ** 2)

    return (wilson_centre - wilson_offset, wilson_centre + wilson_offset)
    # NOTE
    # ((x + a) - (x - a)) / 2
    # = (2a) / 2
    # = a


@numba.jit(
    nopython=True,
    nogil=True,
    fastmath=True,
)
def _overlap_coefficient(data, inds, *args):
    """
    This function calculates the overlap coefficient for a dataset
    and a single set of diseases.

    Note, the user may provide their own weighting function to use
    in the compute_hypergraph method, and may use this as a template
    as the type has to match the type of this function.

    Parameters
    ----------

        data : numpy.array(dtype=numpy.uint8)
            a table indicating whether an individual (rows) is present
            in a node or not (columns).

        inds : numpy.array(dtype=numpy.int8)
            a vector specifying the edge (sets of nodes) to compute the
            overlap coefficient for. Since this is a numba function that is
            JIT compiled for performance, this is a numpy array. The
            edges is encoded as lists of integer indices representing
            nodes connected to the edge.

    Returns
    -------

        numpy.float64 : The calculated overlap coefficient.

        numpy.float64 : The calculated confidence interval in the overlap coefficient.

        numpy.float64 : The denominator used to calculate the overlap coefficient.

    """
    n_diseases = inds.shape[0]
    
    if n_diseases == 1:
        # if there is one disease, return the prevalence
        numerator = 0.0
        for row in range(data.shape[0]):
            if data[row, inds[0]] > 0:
                numerator += 1.0
        denom = float64(data.shape[0])
       
    else:    
        # if there is more than one disease, return the overlap coeffcient.
        numerator = 0.0
        denominator = zeros(shape=(n_diseases))

        for ii in range(data.shape[0]):
            loop_sum = 0
            for jj in range(n_diseases):
                loop_sum += data[ii, inds[jj]]
                denominator[jj] += data[ii, inds[jj]]

            if loop_sum == n_diseases:
                numerator += 1.0

        denom = denominator[0]
        for jj in range(1, n_diseases):
            if denominator[jj] < denom:
                denom = denominator[jj]


    wilson_ci = _wilson_interval(numerator, denom)
    return numerator / denom, wilson_ci, denom

@numba.jit(
    nopython=True,
    nogil=True,
    parallel=True,
    fastmath=True,
)
def _compute_weights(
    data,
    work_list,
    weight_function,
    args,
):
    """
    This function takes a data table and computes the overlap
    coefficient for all combinations spectified in the work list.
    We also calculate the node weight here.

    Parameters
    ----------

        data : numpy.array(dtype=numpy.uint8)
            a table indicating whether an individual (rows) is present
            in a node or not (columns).

        work_list : numpy.array(dtype=numpy.int8)
            a table specifying the edges (sets of nodes) to compute the
            overlap coefficient for. Since this is a numba function that is
            JIT compiled for performance, this is a numpy array. Each row
            corresponds to an edge. Edges are encoded as lists of integer
            indices representing nodes connected to the edge. When the end
            of the edge is reached, the row is filled to the end with
            -1 (since edges can connect to any number of nodes).

        weight_function : callable, jit compiled function
            a function that takes the data and a vector representing an edge
            and returns a weight.

        *args : optional
            optional data to pass to a custom weight function

    Returns
    -------

        numpy.array
            The incidence_matrix, a n_nodes * n_edges matrix such
            that M * W * M.T is the adjacency matrix of the hypergraph, i.e. each
            element of the incidence matrix is a flag indicating whether
            there is a connection between the appropriate node and edge.

        numpy.array
            The edge weight vector, a one dimensional array of length n_edges
            that contains the calculated edge weights.

        numpy.array
            The edge weight confidence interval vector, a one dimensional array of
            length n_edges that contains the calculated variances in the edge weights.

        numpy.array
            The edge weight population vector, a one dimensional array of length n_edges
            that contains the population sizes used in the calculation of the the edge
            weights.

        numpy.array
            The node weight vector, a one dimensional array of length n_nodes
            that contains the calculated node weights.

        numpy.array
            The node weight confidence interval vector, a one dimensional array of
            length n_nodes that contains the calculated variances in the node weights.

        numpy.array
            The node weight population vector, a one dimensional array of length n_nodes
            that contains the population sizes used in the calculation of the the node
            weights.
    """


    incidence_matrix = zeros(shape=(work_list.shape[0], data.shape[1]), dtype=uint8)
    edge_weight = zeros(shape=work_list.shape[0], dtype=float64)
    edge_weight_ci = zeros(shape=(work_list.shape[0], 2), dtype=float64)
    edge_weight_population = zeros(shape=work_list.shape[0], dtype=float64)

    # NOTE (Jim 14/6/2021)
    # There is an absolutely bizarre bug in Numba 0.53.1
    # (already reported: https://github.com/numba/numba/issues/7105)
    # In nopython, parallel=True JIT mode, unless I print some feature
    # of the work list variable the loop starting on line 82 will do
    # nothing. self.edge_weights will be empty and most of the tests in
    # the test suite will fail. The only reasonable response here is to
    # say "what the fuck?!"
    print("Computing ", work_list.shape[0], " edges")
    # Also, if I put this print statement above the intialisation of the
    # incidence matrix and edge weight vector the tests fail again.
    # *shakes head. Numba is broken.

    for index in numba.prange(work_list.shape[0]):

        inds = work_list[index, :]
        inds = inds[inds != -1]
        n_diseases = inds.shape[0]

        # This call to weight_function allows users to define their
        # own weight functions with or without the optional args.
        # :-)
        weight, ci, denom = weight_function(data, inds, *args)

        edge_weight[index] = weight
        edge_weight_ci[index, :] = ci
        edge_weight_population[index] = denom
        for jj in range(n_diseases):
            incidence_matrix[index, inds[jj]] = 1

    node_weight = zeros(shape=data.shape[1], dtype=float64)
    node_weight_ci = zeros(shape=(data.shape[1], 2), dtype=float64)
    node_weight_population = zeros(shape=data.shape[1] , dtype=float64)

    for index in numba.prange(data.shape[1]):
        weight, ci, denom = weight_function(data, array([index]), *args)
        node_weight[index] = weight
        node_weight_ci[index, :] = ci
        node_weight_population[index] = denom

    return (
        incidence_matrix,
        edge_weight,
        edge_weight_ci,
        edge_weight_population,
        node_weight,
        node_weight_ci,
        node_weight_population,
    )


def _bipartite_eigenvector(incidence_matrix, weight):
    '''
    This computest the largest eigenvalue and corresponding
    eigenvector of the bipartite representation of the hypergraph.

    We use the fact that even though the full adjacency matrix will
    not fit in memory for a relatively small hypergraph, the adjacency
    matrix for the bipartite representation is sparse and essentially
    consists of two copies of the incidence matrix. Therefore functions
    that act on sparse matrices are leveraged to do the calculation.


    Parameters
    ----------

        incidence_matrix : numpy.array (dtype=numpy.float64)
            The incidence matrix multiplied by a diagonal matrix of weights.
            ie (matrix.T * diag(weight)).T

        weight : numpy.array(dtype=numpy.float64)
            A vector of edge weights.

    Returns
    -------

         numpy.array(dtype=numpy.float64)
            The calculated eigenvector

         numpy.array(dtype=numpy.float64)
            The calculated uncertainties in the eigenvector elements

    '''
    weighted_matrix = multiply(incidence_matrix, weight.reshape((-1, 1)))
    n_edges, n_nodes = weighted_matrix.shape
    total_elems = n_edges + n_nodes

    adjacency_matrix = lil_matrix((total_elems, total_elems))
    adjacency_matrix[n_nodes:total_elems, 0:n_nodes] = weighted_matrix
    adjacency_matrix[0:n_nodes, n_nodes:total_elems] = weighted_matrix.T

    eig_val, eig_vec = eigsh(adjacency_matrix, k=1)

    return abs(eig_val[0]), abs(eig_vec.reshape(-1))

@numba.jit(
    [numba.float64[:](numba.uint8[:, :], numba.float64[:], numba.float64[:]),
    numba.float64[:](numba.float64[:, :], numba.float64[:], numba.float64[:])],
    nopython=True,
    parallel=True,
    fastmath=False,
)
def _iterate_vector(incidence_matrix, weight, vector):
    '''
    This function performs one Chebyshev iteration to calculate the
    largest eigenvalue and corresponding eigenvector of the either the
    standard or dual hypergraph depending on the orientation of the
    incidence matrix.

    This function calculates M * W * M^T * V whilst setting all the
    diagonal elements of M * W * M^T to zero.

    Parameters
    ----------

        matrix : numpy.array (dtype=numpy.uint8)
            The incidence matrix

        weight : numpy.array(dtype=numpy.float64)
            A vector of weights, which must have the same number of
            elements as the second axis of matrix.

        vector : numpy.array(dtype=numpy.float64)
            The vector to multiply the matrix by.
            Must have the same number of elements as the second axis
            of matrix.

    Returns
    -------

         numpy.array(dtype=numpy.float64)
            The result of matrix * weight * transpose(matrix) * vector with
            diagonal elements of matrix * weight * transpose(matrix) set to zero.
    '''

    # we are calculating [M W M^T - diag(M W M^T)] v
    term_1 = zeros_like(vector)

    # 1) W M^T
    weighted_incidence = zeros_like(incidence_matrix, dtype=float64)
    for i in numba.prange(weighted_incidence.shape[0]):
        for j in range(weighted_incidence.shape[1]):
            weighted_incidence[i, j] += incidence_matrix[i, j] * weight[j]


    # 2) W M^T v
    intermediate = zeros(weighted_incidence.shape[1], dtype=vector.dtype)
    for k in numba.prange(weighted_incidence.shape[1]):
        for j in range(len(vector)):
            intermediate[k] += weighted_incidence[j, k] * vector[j]

    # 3) M W M^T v
    for i in numba.prange(len(vector)):
        for k in range(weighted_incidence.shape[1]):
            term_1[i] += incidence_matrix[i, k] * intermediate[k]

    # 4) diag(M W M^T v) can be done in one step using W M^T from before
    subt = zeros_like(vector)
    for i in numba.prange(len(vector)):
        for k in range(weighted_incidence.shape[1]):
            subt[i] += incidence_matrix[i, k] * weighted_incidence[i, k] * vector[i]

    # 5) subtract one from the other.
    result = zeros_like(vector)
    for i in numba.prange(len(vector)):
        result[i] = term_1[i] - subt[i]


    return result

def _reduced_powerset(iterable, min_set=0, max_set=None):
    """
    This function computes the (potentially) reduced powerset
    of an iterable container of objects.

    By default, the function returns the full powerset of the
    iterable, including the empty set and the full container itself.
    The size of returned sets can be limited using the min_set and
    max_set optional arguments.

    Parameters
    ----------

        iterable : iterable
            A container of objects for wheich to construct the
            (refuced) powerset.

        min_set : int, optional
            The smallest size of set to include in the reduced
            powerset. Default 0.

        max_set : int, optional
            The largest size of set to include in the reduced
            powerset. By default, sets up to len(iterable) are
            included.

    Returns
    -------

         itertools.chain :
            An iterator that iterates through all possible elements
            in the reduced powerset of the input iterable.
    """

    if max_set is None:
        max_set = len(iterable)+1

    return chain.from_iterable(
        combinations(iterable, r) for r in range(min_set, max_set)
    )


##########################################
## Public API here


class Hypergraph(object):

    def __init__(
        self,
        verbose=True
    ):
        """
        Create an instance of a hypergraph object.
        Currently does no computations.

        Parameters
        ----------

            verbose : bool, optional
                When this is set to true the class methods produce additional output to stdout
                (default: True)
        """

        self.incidence_matrix = None
        self.edge_weights = None
        self.node_weights = None
        self.edge_weights_ci = None
        self.node_weights_ci = None
        self.edge_weights_pop = None
        self.node_weights_pop = None
        self.edge_list = None
        self.node_list = None
        self.verbose = verbose

        return

    def compute_hypergraph(
        self,
        data,
        weight_function=_overlap_coefficient,
        *args,
    ):
        """
        Compute the incidence matrix and weights of a weighted, undirected hypergraph.

        The weight function used is the overlap coefficient for edges and the crude
        prevalence for nodes.

        In jupyter, set the number of CPUs to use by executing
        the following before the function call:
        %env NUMBA_NUM_THREADS=n

        Parameters
        ----------

            data : pandas.DataFrame
                A pandas dataframe with rows corresponding to individuals
                and columns to diseases. Entries of the dataframe should be zero
                or one indicating the person does not or does have the disease
                respectively.

            weight_function : callable, optional
                A numba comiled function that computes the hypergraph weights. This
                must be a function that takes two arguments (a numpy array of
                data and a vector representing an edge to compute the weight for)
                and that returns the single edge weight (a numpy.float64).
                By default, a numba compiled function calculating the overlap
                coefficient as the edge weights is used.

            *args : optional data to pass into the weight function. Unused by
                default - may be used for custom weight calculations.

        Sets
        ----

            incidence_matrix : numpy.array
                a n_nodes * n_edges matrix such
                that M * W * M.T is the adjacency matrix of the hypergraph, i.e. each
                element of the incidence matrix is a flag indicating whether
                there is a connection between the appropriate node and edge.

            edge_weights : numpy.array
                The edge weight vector, a one dimensional array of length n_edges
                that contains the calculated edge weights.

            edge_weights_ci : numpy.array
                The edge weight confidence interval vector, a one dimensional array of
                length n_edges that contains the calculated variances in the edge weights.

            node_weights : numpy.array
                The node weight vector, a one dimensional array of length n_nodes
                that contains the calculated node weights.

            node_weights_ci : numpy.array
                The node weight confidence interval vector, a one dimensional array of
                length n_nodes that contains the calculated variances in the node weights.

            edge_list : list
                The edge list, a list of edges each element of which is a tuple of strings
                derived from the names of the pandas.DataFrame columns and indicates the
                order of edges in the incidence matrix. This is required
                because the edge list is randomly shuffled to improve perfomance.

            node_list : list
                A list of nodes derived from the keys of the input pandas dataframe.

        Returns
        -------

            None

        """

        # construct a matrix of flags from the pandas input
        data_array = data.to_numpy().astype(uint8)

        # create a list of edges (numpy array of indices)
        node_list_string = data.keys().tolist()
        node_list = list(range(data_array.shape[1]))
        n_diseases = len(node_list)

        t = time()
        m_data = unique(data_array, axis=0)

        # I'm very grateful to Alex Lee for providing the initial version of this
        # edge pruning solution. Thank you!

        # Alex's solution (fails some tests because it's
        # returning a set of all diseases.
        #unique_co_occurences = [list(where(elem)[0]) for elem in m_data[sum(m_data, axis=1)>=2]]
        #valid_powersets = [
        #    set(itertools.chain(
        #        *[list(itertools.combinations(elem, i)) for i in range(2, len(elem) + 1)]
        #    )) for elem in unique_co_occurences
        #]
        #edge_list = [list(elem) for elem in set.union(*valid_powersets)]

        # my solution
        m_data = m_data[m_data.sum(axis=1) >= 2]
        edge_list = list(set().union(*[
            list(
                _reduced_powerset(
                    where(i)[0],
                    min_set=2,
                    max_set=array([i.sum()+1, n_diseases]).min().astype(int64)
                )
            ) for i in m_data
        ]))
        if self.verbose:
            print("Edge list construction took {:.2f}".format(time() - t))

        # All possible edges solution
        #edge_list = list(
        #    itertools.chain(
        #        *[itertools.combinations(node_list, ii) for ii in range(2, len(node_list))]
        #    )
        #)

        max_edge = array([len(ii) for ii in edge_list]).max()
        work_list = array(
            [list(ii) + [-1] * (max_edge - len(ii)) for ii in edge_list],
            dtype=int8
        )

        # shuffle the work list to improve runtime
        reindex = arange(work_list.shape[0])
        random.shuffle(reindex)
        work_list = work_list[reindex, :]
        edge_list = array(edge_list, dtype="object")[reindex].tolist()

        # compute the weights
        (inc_mat_original,
        edge_weight,
        edge_weight_ci,
        edge_weight_pop,
        node_weight,
        node_weight_ci,
        node_weight_pop) = _compute_weights(
            data_array,
            work_list,
            weight_function,
            args
        ) # the linter is not going to like this...

        # traverse the edge list again to create a list of string labels
        edge_list_out = [[node_list_string[jj] for jj in ii] for ii in edge_list]

        # get rid of rows that are all zero.
        inds = edge_weight > 0
        inc_mat_original = inc_mat_original[inds, :]
        edge_weight = edge_weight[inds]
        edge_weight_ci = edge_weight_ci[inds]
        edge_weight_pop = edge_weight_pop[inds]
        
        edge_list_out = array(edge_list_out, dtype="object")[inds].tolist()
        # traverse the edge list one final time to make sure the edges are tuples
        edge_list_out = [tuple(ii) for ii in edge_list_out]

        self.incidence_matrix = inc_mat_original
        self.edge_weights = edge_weight
        self.edge_weights_ci = edge_weight_ci
        self.edge_weights_pop = edge_weight_pop
        self.node_weights = node_weight
        self.node_weights_ci = node_weight_ci
        self.node_weights_pop = node_weight_pop
        self.edge_list = edge_list_out
        self.node_list = node_list_string
        return

    def eigenvector_centrality(
            self,
            rep="standard",
            weighted_resultant=False,
            tolerance=1e-6,
            max_iterations=100,
            random_seed=12345,
            bootstrap_samples=1,
            bootstrap_randomisation_function=_binomial_rvs,
        ):

        """
        This function uses a Chebyshev algorithm to find the eigenvector
        and largest eigenvalue (corresponding to the eigenvector
        centrality) from the incidence matrix and a weight vector.
        Note, the adjacency matrix is calculated using M * W * C^T, so the
        size of the first dimension of the incidence matrix will be the
        length of the returned eigenvector

        In jupyter, set the number of CPUs to use by executing
        the following before the function call:
        %env NUMBA_NUM_THREADS=n

        One may use this function to compute the centrality of the standard,
        the dual hypergraph or the bipartite represention of the hypergraph

        Examples
        --------

        >>> h = Hypergraph()
        >>> h.eigenvector_centrality()
        computes the eigenvector centrality of the standard hypergraph.

        >>> h.eigenvector_centrality(rep="dual")
        computes the eigenvector centrality of the dual hypergraph.

        >>> h.eigenvector_centrality(rep="bipartite")
        computes the eigenvector centrality of the bipartite representation of the hypergraph.

        Parameters
        ----------
            rep : string, optional
                The representation of the hypergraph for which to calculate the
                eigenvector centrality. Options are "standard", "dual" or "bipartite"
                (default "standard")

            tolerance : float, optional
                The difference between iterations in the eigenvalue at which to
                assume the algorithm has converged (default: 1e-6)

            weighted_resultant : bool, optional
                This flag tells the the algorithm to whether to include both edge and node
                weights in the eigenvector centrality calculation. If this is set to False,
                the eigenvectors of M^T W M are calculated (or equivalent for the dual hypergraph.
                If set to True then the eigenvectors of sqrt(W_a) M^T W_b M sqrt(W_a) are
                calculated. This flag is ignored for the bipartite rep of the hypergraph.

            max_iterations : int, optional
                The maximum number of iterations (of the power method) to perform before
                terminating the algorithm and assuming it has not converged (default: 100)

            random_seed : int, optional
                The random seed to use in generating the initial vector (default: 12345)

            bootstrap_samples : int, optional
                The number of bootstrap samples to use to estimate the uncertainties in the
                eigenvector centrality. (default: 1)

        Returns
        -------

            float
                The calculated largest eigenvalue of the adjacency matrix

            float
                The calculated error in the eigenvalue

            numpy.array
                the eigenvector centrality of the hypergraph. The order of the elements is
                the same as the order passed in via the incidence matrix.

        """

        # Much of this code is inspired by a solution provided to me by Ed Bennett of
        # SA2C in this repo:
        # https://github.com/sa2c/chebysolve
        # Thank you Ed!

        # 0) setup
        rng = random.default_rng(random_seed)

        if rep == "standard":
            inc_mat_original = self.incidence_matrix.T
            old_eigenvector_estimate = rng.random(self.incidence_matrix.shape[1], dtype='float64')

            weight_original = self.edge_weights
            weight_population = self.edge_weights_pop
            weight_resultant = self.node_weights
            resultant_pop = self.node_weights_pop
        elif rep == "dual":
            inc_mat_original = self.incidence_matrix
            old_eigenvector_estimate = rng.random(self.incidence_matrix.shape[0], dtype='float64')

            weight_original = self.node_weights
            weight_population = self.node_weights_pop
            weight_resultant = self.edge_weights
            resultant_pop = self.edge_weights_pop
        elif rep == "bipartite":

            # We treat the bipartite representation differently -
            # the adjacency matrix is sparse, so we can use a sparse matrix
            # datatype and compute the eigevector directly

            eigenvector_boot = []
            eigenvalue_boot = []
            for _ in range(bootstrap_samples):

                inc_mat = self.incidence_matrix
                print(inc_mat.shape)
                if bootstrap_samples > 1:
                    weight = randomize_weights(
                        self.edge_weights_pop.astype(int32), 
                        self.edge_weights, 
                        bootstrap_randomisation_function
                    )
                else:
                    weight = self.edge_weights

                eig_val, eig_vec = _bipartite_eigenvector(
                    inc_mat,
                    weight
                )

                eigenvector_boot.append(eig_vec / eig_vec.sum())
                eigenvalue_boot.append(eig_val)

            eigenvector_boot = array(eigenvector_boot)

            return eigenvector_boot.mean(axis=0), eigenvector_boot.std(axis=0)

        else:
            raise Exception("Representation not supported.")


        # 1) Initial checks

        # Check the weight vector is the right shape
        # not sure this is needd because both of these variables are coming from
        # internal state, but we'll keep it unless we end up with a reason to
        # get rid of it.
        if inc_mat_original.shape[1] != len(weight_original):
            raise Exception(
                ("The weight vector and the second index of the " +
                 "incidence matrix must be the same length")
            )

        # 2) do the Chebyshev

        eigenvector_boot = []
        eigenvalue_boot = []

        for _ in range(bootstrap_samples):

            # apply a perturbation to the weights. Note, we will not do this if the
            # user has requested only 1 bootstrap iteration or if the uncertainties in the
            # weights are set to zero.

            if weighted_resultant:
                res_weight = weight_resultant
            else:
                res_weight = ones_like(weight_resultant)

            if bootstrap_samples > 1: # only perturb the weights if there is more than one sample
                if weighted_resultant:
                    res_weight = randomize_weights(
                        resultant_pop.astype(int32), 
                        res_weight, 
                        bootstrap_randomisation_function
                    )
                weight = randomize_weights(
                    weight_population.astype(int32), 
                    weight_original, 
                    bootstrap_randomisation_function
                )

            else:
                weight = weight_original

            weight_norm = norm(weight)
            res_weight_norm = norm(res_weight)

            # I'm not sure if we actually need to apply the normalisation to the vectors,
            # since the eigenvector is being normalised in the end anyway. I don't think this
            # is using that much time compared to the rest of the function so I will leave it
            # in for now.
            weight = weight / weight_norm
            res_weight = res_weight / res_weight_norm

            inc_mat = diags(sqrt(res_weight)).dot(inc_mat_original)
            old_eigenvector_estimate /= norm(old_eigenvector_estimate)
            eigenvalue_estimates, eigenvalue_error_estimates = [], []

            # In principle, the body of this loop could be compiled with Numba.
            # However, iterate_vector() is quadratic in long_axis_size, whereas
            # all the other operations here are linear in it, so we are spend very
            # little time in the rest of this loop body.
            for iteration in range(max_iterations):

                if self.verbose:
                    print("\rRunning iteration {}...".format(iteration), end="")

                new_eigenvector_estimate = _iterate_vector(
                    inc_mat,
                    weight,
                    old_eigenvector_estimate
                )

                # To estimate eigenvalue, take ratio of new to old eigenvector
                # ignoring zeroes
                mask = (new_eigenvector_estimate != 0) & (old_eigenvector_estimate != 0)
                iter_eigenvalue_estimates = new_eigenvector_estimate[mask] / old_eigenvector_estimate[mask]
                eigenvalue_estimate = iter_eigenvalue_estimates.mean()
                eigenvalue_error_estimate = iter_eigenvalue_estimates.std()

                eigenvalue_estimates.append(eigenvalue_estimate)
                eigenvalue_error_estimates.append(eigenvalue_error_estimate)

                if eigenvalue_error_estimate / eigenvalue_estimate < tolerance:

                    if self.verbose:
                        print(
                            "\nConverged at largest eigenvalue {:.2f} ± {:.4f} after {} iterations".format(
                                eigenvalue_estimate,
                                eigenvalue_error_estimate,
                                iteration
                            )
                        )
                    break

                # Normalise to try to prevent overflows
                old_eigenvector_estimate = (
                    new_eigenvector_estimate /
                    norm(new_eigenvector_estimate)
                )

            else:
                if self.verbose:
                    print("\nFailed to converge after", iteration, "iterations.")
                    print("Last estimate was {:.2f} ± {:.4f}".format(
                        eigenvalue_estimate,
                        eigenvalue_error_estimate
                        )
                    )

            eigenvalue_boot.append(eigenvalue_estimate)
            # we are applying a scaling to the eigenvector here, so in principle the magnitude also has meaning.
            eigenvector_boot.append(weight_norm * res_weight_norm * new_eigenvector_estimate / norm(new_eigenvector_estimate))

        eigenvector_boot = array(eigenvector_boot)
        return eigenvector_boot.mean(axis=0), eigenvector_boot.std(axis=0)


    def degree_centrality(
        self,
        rep="standard",
        weighted=True
    ):
        """
        This method calculates the degree centrality for the hypergraph.

        One may use this function to compute the centrality of the standard or
        the dual hypergraph.

        Examples
        --------

        >>> h = Hypergraph()
        >>> h.degree_centrality()
        computes the weighted degree centrality of the standard hypergraph.

        >>> h.degree_centrality(rep="dual")
        computes the weighted degree centrality of the dual hypergraph.

        >>> h.degree_centrality(weighted=False)
        computes the unweighted (i.e. all edges are considered to have weight=1)
        degree centrality of the standard hypergraph.

        Parameters
        ----------

            rep : string, optional
                The representation of the hypergraph for which to calculate the
                eigenvector centrality. Options are "standard" or "dual"
                (default "standard")

            weighted : bool, optional
                Whether or not to use the weights to calculate the degree
                centrality (default: True)


        Returns
        -------

            list
                The calculated degree centralities
        """

        M = self.incidence_matrix
        if rep == "standard":
            ax = 0
            if weighted:
                M = diags(self.edge_weights).dot(self.incidence_matrix)

        elif rep == "dual":
            ax = 1
            if weighted:
                M = csr_matrix(self.incidence_matrix).dot(diags(self.node_weights))

        else:
            raise Exception("Representation not supported.")

        return list((M.sum(axis=ax)).flat)

if __name__ == "__main__":

    n_people = 5000
    n_diseases = 10

    import pandas as pd
    import numpy as np

    data = (np.random.rand(n_people, n_diseases) > 0.8).astype(np.uint8)
    data_pd = pd.DataFrame(
        data
    ).rename(
        columns={i: "disease_{}".format(i) for i in range(data.shape[1])}
    )

    h = Hypergraph(verbose=False)
    h.compute_hypergraph(data_pd)

    e_vec, e_vec_err = h.eigenvector_centrality(
        rep="dual",
        weighted_resultant=True,
        bootstrap_samples=10
    )

    print(np.linalg.norm(e_vec))
    print()
    print(h.edge_weights[0])
    print(h.edge_weights_pop[0])
    print()
    print(e_vec[0])
    print(e_vec_err[0])
    print()
    print(randomize_weights(np.ones(10) * h.edge_weights_pop[0], np.ones(10) * h.edge_weights[0]))



