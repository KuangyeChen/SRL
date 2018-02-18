from sklearn.metrics import auc, precision_recall_curve, average_precision_score, precision_score
import numpy as np

__all__ = ['make_corrupt_for_train',
           'make_corrupt_for_eval',
           'make_split',
           'metrics_in_a_batch']


def make_corrupt_for_train(one_batch, database, num_entities, corrupt_size):
    """
    Make corrupted triples for training.
    Only corrupt e2 in (e1, e2, r).

    Final structure is:
        batch_data: [True(1),    True(1),    ... , True(1),               True(2),  ...]
      corrupted_e2: [Fake(1)(1), Fake(1)(1), ... , Fake(1)(corrupt_size), Fake(2)(1), ...]

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


def make_split(one_batch, extra_data, num_relations, make_zero_in_extra_list=False):
    """
    Split data into corresponding relation.
    Final structure:
        [[all triples for relation_1], [all triples for relation_2], ... ]

    :param one_batch: To split data batch
    :param extra_data: Some extra data needed. e.g. labels, corrupted e2...
    :param num_relations: Number of relationship
    :param make_zero_in_extra_list: If extra data also must not be empty.
    :return: batch_list: Holding lists of triples.
             extra_list: Holding lists of extra data.
             empty_r_list: Indicating if relation_r's List is empty.
    """

    batch_list = [[] for _ in range(num_relations)]
    extra_list = [[] for _ in range(num_relations)]
    for i in range(len(one_batch)):
        batch_list[one_batch[i][2]].append((one_batch[i][0], one_batch[i][1]))
        extra_list[one_batch[i][2]].append(extra_data[i])
    empty_r_list = [len(batch_in_r) == 0 for batch_in_r in batch_list]
    batch_list = [batch_in_r if len(batch_in_r) != 0 else [(0, 0)] for batch_in_r in batch_list]
    if make_zero_in_extra_list:
        extra_list = [extra_in_r if len(extra_in_r) != 0 else [extra_data[0]] for extra_in_r in extra_list]
    return batch_list, extra_list, empty_r_list


def make_corrupt_for_eval(one_batch, database, num_entities, corrupt_size):
    """
    Make corrupted triples for evaluation.
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


def metrics_in_a_batch(predicts, labels):
    """
    Calculate the metrics:
        Mean Reciprocal Rank
        Hit at 10
        Area under precision recall curve
        Average precision score
        (temporary) Precision for threshold 0.5

    :param predicts: Predicts for each triple.
    :param labels: True labels for each triple.
    :return: MRR, HIT@10, AUC_PR, AP, PRC_0.5.
    """

    predicts_list = np.split(predicts, np.squeeze(np.argwhere(labels == 1))[1:])
    # print(predicts_list[0])
    # print(len(predicts_list[0]) - predicts_list[0].argsort().argsort()[0])
    mrr = 1 / np.array([len(one_predict) - one_predict.argsort().argsort()[0] for one_predict in predicts_list])
    mrr = mrr.mean()
    hit_at_10 = np.array([one_predict.argsort().argsort()[0] >= len(one_predict) - 10 for one_predict in predicts_list])
    hit_at_10 = hit_at_10.mean()
    prec_zero = precision_score(labels, predicts>0.5)
    prec, recall, _ = precision_recall_curve(labels, predicts)
    return [mrr, hit_at_10, auc(recall, prec), average_precision_score(labels, predicts), prec_zero]