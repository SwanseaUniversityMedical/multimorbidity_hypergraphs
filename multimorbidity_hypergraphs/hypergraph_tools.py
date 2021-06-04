"""
This is a module of tools for use in analysing 2D tables of boolean data
(my use case is multimorbidity, so rows represent people and columns represent
the diseases they do or do not have) by constructing and analysing hypergraph
data structures.
"""


import itertools
import numpy as np
import numba

@numba.jit(
        numba.types.Tuple((numba.uint8[:, :], numba.float64[:], numba.float64[:])) # outputs
        (numba.uint8[:, :], numba.int8[:, :]), # inputs
    nopython=True,
    nogil=True,
    parallel=True,
    fastmath=True,
)
def overlap_coefficient_numba(data, work_list):
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
            The node weight vector, a one dimensional array of length n_nodes
            that contains the calculated node weights.
    """

    incidence_matrix = np.zeros(shape=(work_list.shape[0], work_list.shape[1] + 1), dtype=np.uint8)
    edge_weight = np.zeros(shape=work_list.shape[0], dtype=np.float64)

    for index in numba.prange(work_list.shape[0]):

        inds = work_list[index, :]
        inds = inds[inds != -1]
        n_diseases = inds.shape[0]

        numerator = 0.0
        denominator = np.zeros(shape=(n_diseases))

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

        weight = numerator / denom
        edge_weight[index] = weight
        for jj in range(n_diseases):
            incidence_matrix[index, inds[jj]] = 1

    node_weight = np.zeros(shape=data.shape[1], dtype=np.float64)

    for index in numba.prange(data.shape[1]):
        numerator = 0.0
        for row in range(data.shape[0]):
            if data[row, index] > 0:
                numerator += 1.0
        node_weight[index] = (numerator / np.float64(data.shape[0])) ** 2


    return (incidence_matrix, edge_weight, node_weight)

def compute_hypergraph(data):
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
            The node weight vector, a one dimensional array of length n_nodes
            that contains the calculated node weights.

        list
            The edge list, a list of edges each element of which is a tuple of strings
            derived from the names of the pandas.DataFrame columns and indicates the
            order of edges in the incidence matrix. This is required
            because the edge list is randomly shuffled to improve perfomance.

    """

    # construct a matrix of flags from the pandas input
    data_array = data.to_numpy().astype(np.uint8)

    # create a list of edges (numpy array of indices)
    node_list_string = data.keys().tolist()
    node_list = list(range(data_array.shape[1]))
    edge_list = list(
        itertools.chain(
            *[itertools.combinations(node_list, ii) for ii in range(2, len(node_list))]
        )
    )
    max_edge = np.max([len(ii) for ii in edge_list])
    work_list = np.array(
        [list(ii) + [-1] * (max_edge - len(ii)) for ii in edge_list],
        dtype=np.int8
    )

    # shuffle the work list to improve runtime
    reindex = np.arange(work_list.shape[0])
    np.random.shuffle(reindex)
    work_list = work_list[reindex, :]
    edge_list = np.array(edge_list)[reindex].tolist()


    # compute the weights
    (inc_mat, edge_weight, node_weight) = overlap_coefficient_numba(data_array, work_list)
    # traverse the edge list again to create a list of string labels
    edge_list_out = [tuple([node_list_string[jj] for jj in ii]) for ii in edge_list]

    # get rid of rows that are all zero.
    inds = edge_weight > 0
    inc_mat = inc_mat[inds, :]
    edge_weight = edge_weight[inds]
    edge_list_out = np.array(edge_list_out)[inds].tolist()

    return inc_mat, edge_weight, node_weight, edge_list_out


@numba.jit(
    numba.float64[:](numba.float64[:, :], numba.float64[:]),
    nopython=True,
    parallel=True,
    fastmath=True,
)
def __iterate_vector_bipartite(matrix, vector):
    '''
    This function performs one Chebyshev iteration to calculate the
    largest eigenvalue and corresponding eigenvector of the bipartite
    representation of the hypergraph.

    Note: the variable ``matrix'' here must include the weights, since the
    adjacency matrix is constructed by placing this matrix in the appropriate
    place in a matrix of zeros.

    Parameters
    ----------

        matrix : numpy.array (dtype=numpy.float64)
            The incidence matrix multiplied by a diagonal matrix of weights.
            ie (matrix.T * diag(weight)).T

        vector : numpy.array(dtype=numpy.float64)
            The vector to multiply the (bipartite adjacency) matrix by.
            Must have n_nodes + n_edges elements.

    Returns
    -------

         numpy.array(dtype=numpy.float64)
            The result of matrix * vector

    '''

    n_edges, n_nodes = matrix.shape
    result = np.zeros_like(vector)
    for i in numba.prange(len(vector)):

        if i < n_nodes:
            j_lim = range(n_nodes, n_edges + n_nodes)
        else:
            j_lim = range(0, n_nodes)


        for j in j_lim:

            if i >= n_nodes and j < n_nodes:
                result[i] += matrix[i - n_nodes, j] * vector[j]
            elif i < n_nodes and j >= n_nodes and j <= (n_edges + n_nodes):
                result[i] += matrix[j - n_nodes, i] * vector[j]

        # treat the matrix as if it has ones on the diagonal.
        # This is because this matrix is guaranteed(?) to be singular and symmetric, therefore
        # the eigenvalues form +/- pairs and there is no single eigenvalue with the largest
        # modulus. Without this the algorithm will never converge
        result[i] += vector[i]


    return result

@numba.jit(
    numba.float64[:](numba.uint8[:, :], numba.float64[:], numba.float64[:]),
    nopython=True,
    parallel=True,
    fastmath=True,
)
def __iterate_vector(incidence_matrix, weight, vector):
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

    result = np.zeros_like(vector)
    intermediate = np.zeros_like(incidence_matrix, dtype=np.float64)

    # this bit is M * W where W = diag(weights)
    for i in numba.prange(intermediate.shape[0]):
        for j in range(intermediate.shape[1]):
            intermediate[i, j] += incidence_matrix[i, j] * weight[j]

    for i in numba.prange(len(vector)):
        for j in range(len(vector)):
            if i != j:
                for k in range(incidence_matrix.shape[1]):
                    result[i] += intermediate[i, k] * incidence_matrix[j, k] * vector[j]

    return result

def eigenvector_centrality(
        incidence_matrix,
        weight,
        tolerance=1e-6,
        max_iterations=100,
        random_seed=12345,
        bipartite=False,
        verbose=False,
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

    >>> eigenvector_centrality(inc_mat.T, edge_weight)
    computes the eigenvector centrality of the standard hypergraph.

    >>> eigenvector_centrality(inc_mat, node_weight)
    computes the eigenvector centrality of the dual hypergraph.

    >>> eigenvector_centrality(inc_mat, node_weight, bipartite=True)
    computes the eigenvector centrality of the bipartite representation of the hypergraph.

    Parameters
    ----------
        incidence_matrix : numpy.array
            The incidence matrix (M) of the hypergraph. this is an n x m
            matrix such that M^T * W * M is the adjacency matrix. Can use
            the output of compute_hypergraph here

        weight : numpy.array
            A vector corresponding to the weights of the hypergraph. The length
            of this vector must be equal to the length of the second axis
            of the incidence matrix (since the adjacency matrix is calculated
            as M^T * W * M). This function can be used with unweighted hypergraphs
            by passing in an array of ones of the appropriate length.

        tolerance : float, optional
            The difference between iterations in the eigenvalue at which to
            assume the algorithm has converged (default: 1e-6)

        max_iterations : int, optional
            The maximum number of iterations to perform before terminating the
            algorithm and assuming it has not converged (default: 100)

        random_seed : int, optional
            The random seed to use in generating the initial vector (default: 12345)

        bipartite : bool, optional
            Whether or not to compute the eigenvector centrality of the bipartite representation
            of the hypergraph. This allows one to compute a centrality for the nodes and edges
            of the graph together. (default: False)

        verbose : bool, optional
            When this is set to true the function produces additional output to stdout
            (default: False)

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
    # 1) Check we don't have an adjacency matrix by checking the size of incidence matrix

    long_axis_size, short_axis_size = np.shape(incidence_matrix)
    if long_axis_size == short_axis_size:
        # What's the probability of having the same number of nodes and edges by chance?!
        raise Exception(
            "This function must be called on the incidence matrix, not the adjacency matrix"
        )

    # Check the incidence matrix is the right datatype
    if incidence_matrix.dtype != np.uint8:
        if verbose:
            print("Converting incidence matrix type to u8")
        incidence_matrix = incidence_matrix.astype(np.uint8)

    # Check the weight vector is the right shape
    if short_axis_size != len(weight):
        raise Exception(
            ("The weight vector and the second index of the "+
             "incidence matrix must be the same length")
        )

    rng = np.random.default_rng(random_seed)

    # 2) do the Chebyshev


    # Normalised random initial vector
    if bipartite:
        old_eigenvector_estimate = rng.random(long_axis_size + short_axis_size, dtype='float64')
        incidence_matrix = np.dot(incidence_matrix.T, np.diag(weight)).T
    else:
        old_eigenvector_estimate = rng.random(long_axis_size, dtype='float64')
    old_eigenvector_estimate /= np.linalg.norm(old_eigenvector_estimate)

    eigenvalue_estimates, eigenvalue_error_estimates = [], []

    # In principle, the body of this loop could be compiled with Numba.
    # However, iterate_vector() is quadratic in long_axis_size, whereas
    # all the other operations here are linear in it, so we are spend very
    # little time in the rest of this loop body.

    for iteration in range(max_iterations):

        if verbose:
            print("\rRunning iteration {}...".format(iteration), end="")

        if bipartite:
            new_eigenvector_estimate = __iterate_vector_bipartite(
                incidence_matrix,
                old_eigenvector_estimate
            )
        else:
            new_eigenvector_estimate = __iterate_vector(
                incidence_matrix,
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

            if verbose:
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
            np.linalg.norm(new_eigenvector_estimate)
        )

    else:
        if verbose:
            print("\nFailed to converge after", iteration, "iterations.")
            print("Last estimate was {:.2f} ± {:.4f}".format(
                eigenvalue_estimate,
                eigenvalue_error_estimate
                )
            )

    if bipartite:
        eigenvalue_estimate = eigenvalue_estimate - 1.0

    return (
        eigenvalue_estimate,
        eigenvalue_error_estimate,
        new_eigenvector_estimate
    )


if __name__ == "__main__":

    print("this is here for testing purposes only")
