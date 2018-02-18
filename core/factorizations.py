import numpy as np
from scipy.sparse import csr_matrix
from core.sktensor import dtensor, tucker_hooi, cp_als
from core.sktensor.rescal import als as rescal_als
from core.link_predict_utils import make_split

__all__ = ['rescal_eval',
           'make_sparse_matrix_for_rescal',
           'rescal',
           'make_tensor_from_triple_list',
           'tucker',
           'cp']


def rescal_eval(a_matrix, r_tensor, triple):
    """
    Evaluate the confidence value of a triple using rescal result.

    :param a_matrix: Matrix A in rescal.
    :param r_tensor: Tensor R in rescal.
    :param triple: To evaluate triple.
    :return: The confidence value.
    """

    return np.matmul(a_matrix[triple[0]], np.matmul(r_tensor[triple[2]], a_matrix[triple[1]]))


def make_sparse_matrix_for_rescal(data, num_entities, num_relations):
    """
    Make sparse matrix for each relations.
    Preprocessing for Rescal.

    :param data: List of triples.
    :param num_entities: Number of entities.
    :param num_relations: Number of relations.
    :return: A list of sparse matrix.
    """
    train_r_list, _, empty_r_list = make_split(data, [None] * len(data), num_relations)

    matrix_list = []
    for r in range(num_relations):
        if empty_r_list[r]:
            tmp = csr_matrix((num_entities, num_entities))
        else:
            tmp = csr_matrix(([1] * len(train_r_list[r]), zip(*train_r_list[r])),
                             shape=[num_entities, num_entities])
        matrix_list.append(tmp)
    return matrix_list


def rescal(tensor, rank):
    """
    Make a predict on a database using Rescal.

    :param tensor: list of sparse matrix containing the database.
                   Typically returned by make_sparse_matrix_for_rescal.
    :param rank: used in Rescal algorithm.
    :return:
        predict: a tensor containing predict result.
    """

    if rank >= tensor[0].shape[0] or rank <= 0:
        raise ValueError('Rank should be in (0, %d)' % tensor[0].shape[0])

    a_matrix, r_tensor, _, _, _ = rescal_als(tensor, rank,
                                             init='nvecs',
                                             lambda_A=10, lambda_R=10)
    return a_matrix, r_tensor


def make_tensor_from_triple_list(data, num_entities, num_relations):
    """
    Make a tensor from a list of triples.

    :param data: List of triples.
    :param num_entities: Number of entities.
    :param num_relations: Number of relations.
    :return: A Tensor (numpy array)
    """
    array = np.zeros([num_entities, num_entities, num_relations])
    for triple in data:
        array[triple[0], triple[1], triple[2]] = 1
    return array


def tucker(tensor, rank):
    """
    Make a predict on a database using Tucker.

    :param tensor: Contains the database.
    :param rank: Used in Tucker algorithm.
    :return:
        predict: A tensor containing predict result.
    """

    if len(rank) != tensor.ndim:
        raise ValueError('Tucker requires rank for each dimension of tensor')
    for i in range(len(rank)):
        if rank[i] >= tensor.shape[i] or rank[i] <= 0:
            raise ValueError('Rank should be in [(0, %d), (0, %d), (0, %d)]' % (
                tensor.shape[0], tensor.shape[1], tensor.shape[2]))

    core, fac = tucker_hooi(dtensor(tensor), rank)
    predict = core.ttm(fac[0], 0).ttm(fac[1], 1).ttm(fac[2], 2)
    return predict


def cp(tensor, rank):
    """
    Make a predict on a database using CP.

    :param tensor: Contains the database.
    :param rank: Used in CP algorithm.
    :return:
        predict: A tensor containing predict result.
    """

    if rank >= tensor.shape[2] or rank <= 0:
        raise ValueError('Rank should be in (0, %d)' % tensor.shape[2])
    fac, _, _, _ = cp_als(dtensor(tensor), rank)
    return fac.totensor()
