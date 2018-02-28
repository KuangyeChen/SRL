import numpy as np
import collections
from sklearn.metrics import auc, precision_recall_curve, average_precision_score, precision_score
from typing import Sequence, List, Tuple
from core.knowledge_graph import *

__all__ = ['make_corrupt',
           'make_split',
           'metrics_in_a_batch',
           'report_metrics',
           'Metrics']

Metrics = collections.namedtuple('metrics', ['mrr',
                                             'hit_at_10',
                                             'hit_at_5',
                                             'hit_at_1',
                                             'tp_fp_sum',
                                             'precision',
                                             'auc_pr',
                                             'ap'])


def make_split(one_batch, labels, num_relations):
    """
    Split data into corresponding relation.
    Final structure:
        [[all triples for relation_1], [all triples for relation_2], ... ]

    :param Sequence[Triple] one_batch: To split data batch
    :param Sequence[int] labels: Labels
    :param int num_relations: Number of relationship
    :return: batch_list: Holding lists of triples.
             extra_list: Holding lists of extra data.
             empty_r_list: Indicating if relation_r's List is empty.
    """

    batch_list = [[] for _ in range(num_relations)]                     # type: List[List[Tuple[int, int]]]
    labels_list = [[] for _ in range(num_relations)]                    # type: List[List[int]]
    for i in range(len(one_batch)):
        batch_list[one_batch[i][2]].append((one_batch[i][0], one_batch[i][1]))
        labels_list[one_batch[i][2]].append(labels[i])
    r_empty_list = [len(batch_in_r) == 0 for batch_in_r in batch_list]  # type: List[List[bool]]
    batch_list = [batch_in_r if len(batch_in_r) != 0 else [(0, 0)] for batch_in_r in batch_list]
    rearrange_labels = np.hstack(labels_list)                           # type: np.ndarray

    return batch_list, rearrange_labels, r_empty_list


def make_corrupt(one_batch, database, corrupt_size):
    """
    Make corrupted triples.
    Final structure is:
        [True(1), Fake(1)(1), Fake(1)(2), ... , Fake(1)(corrupt_size), True(2), Fake(2)(1), ...]

    :param Sequence[Triple] one_batch: To corrupt data batch.
    :param KnowledgeGraph database: Knowledge graph with all data.
    :param int corrupt_size: Number of corrupted triples.
    :return: batch_data: List holding triples.
             labels: List holding labels.
    """

    num_entities = database.number_of_entities()
    batch_data = []                                                     # type: List[Triple]
    labels = []                                                         # type: List[int]

    for triple in one_batch:
        batch_data.append(triple)
        labels.append(1)

        all_idx = np.random.permutation(num_entities)

        idx = 0
        for corrupt_i in range(corrupt_size):
            while idx < len(all_idx) and database.check_triple((triple[0], all_idx[idx], triple[2])):
                idx = idx + 1
            if idx == len(all_idx):
                break

            batch_data.append((triple[0], all_idx[idx], triple[2]))
            labels.append(0)
            idx = idx + 1

    return batch_data, labels


def metrics_in_a_batch(predicts, labels, threshold=0.5):
    """
    Calculate the metrics:
        Mean Reciprocal Rank
        Hit at 10
        Hit at 5
        Hit at 1
        Area under precision recall curve
        Average precision score
        Precision at threshold

    :param np.ndarray predicts: Predicts for each triple.
    :param np.ndarray labels: True labels for each triple.
    :param float threshold: Threshold for decision.
    :return: A Metrics handle for all metrics
    """

    if type(predicts) != np.ndarray or type(labels) != np.ndarray:
        raise TypeError("Predicts and labels should be numpy ndarray")

    predicts_list = np.split(predicts, np.squeeze(np.argwhere(labels == 1))[1:])

    mrr = []                                                            # type: List[float]
    hit_at_10 = []                                                      # type: List[bool]
    hit_at_5 = []                                                       # type: List[bool]
    hit_at_1 = []                                                       # type: List[bool]
    for one_group in predicts_list:
        ranks = np.empty_like(one_group, dtype=np.int32)
        ranks[one_group.argsort()] = len(one_group) - np.arange(len(one_group))
        mrr.append(1 / ranks[0])
        hit_at_10.append(ranks[0] <= 10)
        hit_at_5.append(ranks[0] <= 5)
        hit_at_1.append(ranks[0] == 1)

    pos_predict = predicts > threshold
    precision = precision_score(labels, pos_predict)
    tp_fp_sum = sum(pos_predict)

    precision_list, recall_list, _ = precision_recall_curve(labels, predicts)

    return Metrics(np.mean(mrr), np.mean(hit_at_10), np.mean(hit_at_5), np.mean(hit_at_1),
                   tp_fp_sum, precision,
                   auc(recall_list, precision_list),
                   average_precision_score(labels, predicts))


def report_metrics(metrics):
    """
    Print metrics.

    :param Metrics metrics: All metrics
    """

    print("MRR: %.4f, HIT@10: %.4f, HIT@5: %.4f, HIT@1: %.4f" % (
        metrics.mrr, metrics.hit_at_10, metrics.hit_at_5, metrics.hit_at_1))
    print("AUC_PR: %.4f, Average Precision: %.4f" % (metrics.auc_pr, metrics.ap))
    print('TP+FP@0.5: %d, Precision@0.5: %.4f' % (metrics.tp_fp_sum, metrics.precision))
