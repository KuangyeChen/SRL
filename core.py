import re
import numpy as np
from scipy.sparse import lil_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import auc, roc_auc_score
from sktensor import dtensor, tucker_hooi, cp_als
from sktensor.rescal import als as rescal_als

__all__ = ['parse_triple',
           'read_database_from_file',
           'predict_rescal_als',
           'predict_tucker_hooi',
           'predict_cp_als',
           'validation_in_tensor']


def parse_triple(a_triple):
    """
    Parse a relation triple.
    A triple must have form: 'R(relation,entity,entity)'
    e.g. 'R(Economicaid,China,Egypt)'

    :param a_triple: a string of a triple.
    :return:
        triple_lst: a list of strings.
            containing three words of the triple.
            e.g. ['Economicaid', 'China', 'Egypt']
    """

    triple_lst = re.split(r'[(,)]', a_triple)
    if len(triple_lst) != 5 or triple_lst[0] != 'R':
        raise ValueError('a triple must have form: R(relation,entity,entity)')

    return triple_lst[1:4]


def read_database_from_file(filename):
    """
    Read a database from a file.

    :param filename: database file name.
    :return:
        tensor: contains the database.
        rel_names: list containing all relation names.
        ent_names: list containing all entity names.
    """

    rel_names = []
    ent_names = []
    relations = []

    fin = open(filename, 'r')
    for line in fin:
        triple_lst = parse_triple(line)
        if triple_lst[0] not in rel_names:
            rel_names.append(triple_lst[0])
        if triple_lst[1] not in ent_names:
            ent_names.append(triple_lst[1])
        if triple_lst[2] not in ent_names:
            ent_names.append(triple_lst[2])

        relations.append([ent_names.index(triple_lst[1]),
                          ent_names.index(triple_lst[2]),
                          rel_names.index(triple_lst[0])])

    tensor = np.zeros([len(ent_names), len(ent_names), len(rel_names)])
    for rel in relations:
        tensor[rel[0], rel[1], rel[2]] = 1

    fin.close()
    return tensor, rel_names, ent_names


def predict_rescal_als(tensor, rank):
    """
    Make a predict on a database using Rescal.

    :param tensor: contains the database.
    :param rank: used in Rescal algorithm.
    :return:
        predict: a tensor containing predict result.
    """
    if rank >= tensor.shape[0] or rank <= 0:
        raise ValueError('Rank should be in (0, %d)' % tensor.shape[0])

    tensor_slice = [lil_matrix(tensor[:, :, i]) for i in range(tensor.shape[2])]

    A, R, _, _, _ = rescal_als(
        tensor_slice, rank, init='nvecs', conv=1e-3,
        lambda_A=10, lambda_R=10
    )

    r_tensor = np.ndarray(shape=[rank, rank, len(R)])
    for i in range(len(R)):
        r_tensor[:, :, i] = R[i]
    r_tensor = dtensor(r_tensor)
    predict = r_tensor.ttm(A, 0).ttm(A, 1)

    return predict


def predict_tucker_hooi(tensor, rank):
    """
    Make a predict on a database using Tucker.

    :param tensor: contains the database.
    :param rank: used in Tucker algorithm.
    :return:
        predict: a tensor containing predict result.
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


def predict_cp_als(tensor, rank):
    """
    Make a predict on a database using CP.

    :param tensor: contains the database.
    :param rank: used in CP algorithm.
    :return:
        predict: a tensor containing predict result.
    """
    if rank >= tensor.shape[2] or rank <= 0:
        raise ValueError('Rank should be in (0, %d)' % tensor.shape[2])

    fac, _, _, _ = cp_als(dtensor(tensor), rank)
    return fac.totensor()


def validation_in_tensor(tensor, mask_idx, target_idx, func, rank):
    """
    Validation inside a tensor.
    Replace tensor[mask_idx] with 0, and validate on tensor[target_idx]

    :param tensor: contains the database
    :param mask_idx: index of items to replace with 0
    :param target_idx: index of items to validate
    :param func: name of the algorithm to use
                 support 'rescal' 'tucker' 'cp'
    :param rank: used in the algorithm
    :return: AUC of PR-curve, Average precision, AUC of ROC-curve
    """

    algorithm = {'rescal': predict_rescal_als,
                 'tucker': predict_tucker_hooi,
                 'cp': predict_cp_als}

    if func not in algorithm:
        raise ValueError('only support Rescal, Tucker and CP')

    if len(rank) == 1:
        rank = rank[0]

    tensor_for_train = tensor.copy()
    mask_idx = np.unravel_index(mask_idx, tensor.shape)
    target_idx = np.unravel_index(target_idx, tensor.shape)

    # set values to be predicted to zero
    for i in range(len(mask_idx[0])):
        tensor_for_train[mask_idx[0][i], mask_idx[1][i], mask_idx[2][i]] = 0

    # predict unknown values
    predict = algorithm[func](tensor_for_train, rank)

    # compute criteria
    prec, recall, thresholds = precision_recall_curve(tensor[target_idx], predict[target_idx])
    average_prec = average_precision_score(tensor[target_idx], predict[target_idx])
    auc_roc = roc_auc_score(tensor[target_idx], predict[target_idx])

    return auc(recall, prec), average_prec, auc_roc
