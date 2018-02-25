from sklearn.metrics import auc, precision_recall_curve, average_precision_score, precision_score
import numpy as np
import collections

__all__ = ['make_corrupt',
           'make_split',
           'metrics_in_a_batch',
           'report_metrics']


def make_corrupt_for_train(one_batch, database, num_entities, corrupt_size):
    """
    Make corrupted triples for training.
    Only corrupt e2 in (e1, e2, r).

    Final structure is:
        batch_data: [True(1),    True(1),    ... , True(1),               True(2),  ...]
      corrupted_e2: [Fake(1)(1), Fake(1)(2), ... , Fake(1)(corrupt_size), Fake(2)(1), ...]

    :param one_batch: To corrupt data batch
    :param database: Knowledge graph with all data.
    :param num_entities: Number of entities.
    :param corrupt_size: Number of corrupted triples.
    :return: batch_data: List holding triples.
           corrupted_e2: List holding corrupted e2.
    """

    batch_data = []
    corrupted_e2 = []

    for triple in one_batch:
        all_idx = np.random.permutation(num_entities)
        idx = 0
        for corrupt_i in range(corrupt_size):
            while idx < num_entities and database.check_triple(triple[0], all_idx[idx], triple[2]):
                idx = idx + 1
            if idx == len(all_idx):
                break

            batch_data.append((triple[0], triple[1], triple[2]))
            corrupted_e2.append(all_idx[idx])
            idx = idx + 1

    return batch_data, corrupted_e2


def make_split(one_batch, labels, num_relations):
    """
    Split data into corresponding relation.
    Final structure:
        [[all triples for relation_1], [all triples for relation_2], ... ]

    :param one_batch: To split data batch
    :param labels: Labels
    :param num_relations: Number of relationship
    :return: batch_list: Holding lists of triples.
             extra_list: Holding lists of extra data.
             empty_r_list: Indicating if relation_r's List is empty.
    """

    batch_list = [[] for _ in range(num_relations)]
    labels_list = [[] for _ in range(num_relations)]
    for i in range(len(one_batch)):
        batch_list[one_batch[i][2]].append((one_batch[i][0], one_batch[i][1]))
        labels_list[one_batch[i][2]].append(labels[i])
    r_empty_list = [len(batch_in_r) == 0 for batch_in_r in batch_list]
    batch_list = [batch_in_r if len(batch_in_r) != 0 else [(0, 0)] for batch_in_r in batch_list]
    rearrange_labels = np.hstack(labels_list)

    return batch_list, rearrange_labels, r_empty_list


def make_corrupt(one_batch, database, num_entities, corrupt_size):
    """
    Make corrupted triples.
    Final structure is:
        [True(1), Fake(1)(1), Fake(1)(2), ... , Fake(1)(corrupt_size), True(2), Fake(2)(1), ...]

    :param one_batch: To corrupt data batch.
    :param database: Knowledge graph with all data.
    :param num_entities: Number of entities.
    :param corrupt_size: Number of corrupted triples.
    :return: batch_data: List holding triples.
             labels: List holding labels.
    """

    batch_data = []
    labels = []

    for triple in one_batch:
        batch_data.append(triple)
        labels.append(1)

        all_idx = np.random.permutation(num_entities)
        idx = 0
        for corrupt_i in range(corrupt_size):
            while idx < num_entities and database.check_triple(triple[0], all_idx[idx], triple[2]):
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

    :param predicts: Predicts for each triple.
    :param labels: True labels for each triple.
    :param threshold: Threshold for decision.
    :return: A namedtuple containing all metrics
    """

    if type(predicts) != np.ndarray or type(labels) != np.ndarray:
        raise TypeError("Predicts and labels should be numpy ndarray")

    predicts_list = np.split(predicts, np.squeeze(np.argwhere(labels == 1))[1:])

    mrr = 1 / np.array([len(one_predict) - one_predict.argsort().argsort()[0]
                        for one_predict in predicts_list])
    mrr = mrr.mean()

    hit_at_10 = np.array([one_predict.argsort().argsort()[0] >= len(one_predict) - 10
                          for one_predict in predicts_list])
    hit_at_10 = hit_at_10.mean()

    hit_at_5 = np.array([one_predict.argsort().argsort()[0] >= len(one_predict) - 5
                         for one_predict in predicts_list])
    hit_at_5 = hit_at_5.mean()

    hit_at_1 = np.array([one_predict.argsort().argsort()[0] >= len(one_predict) - 1
                         for one_predict in predicts_list])
    hit_at_1 = hit_at_1.mean()

    pos_predict = predicts > threshold
    precision = precision_score(labels, pos_predict)
    tp_fp_sum = sum(pos_predict)

    precision_list, recall_list, _ = precision_recall_curve(labels, predicts)

    metrics = collections.namedtuple('metrics', ['mrr',
                                                 'hit_at_10',
                                                 'hit_at_5',
                                                 'hit_at_1',
                                                 'tp_fp_sum',
                                                 'precision',
                                                 'auc_pr',
                                                 'ap'])

    return metrics(mrr, hit_at_10, hit_at_5, hit_at_1,
                   tp_fp_sum, precision,
                   auc(recall_list, precision_list),
                   average_precision_score(labels, predicts))


def report_metrics(metrics):
    """
    Print metrics.

    :param metrics: A namedtuple containing all metrics
    :return: None
    """

    print("MRR: %.4f, HIT@10: %.4f, HIT@5: %.4f, HIT@1: %.4f" % (
        metrics.mrr, metrics.hit_at_10, metrics.hit_at_5, metrics.hit_at_1))
    print("AUC_PR: %.4f, Average Precision: %.4f" % (metrics.auc_pr, metrics.ap))
    print('TP+FP@0.5: %d, Precision@0.5: %.4f' % (metrics.tp_fp_sum, metrics.precision))
