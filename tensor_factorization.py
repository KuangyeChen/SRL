#!/usr/bin/env python

import numpy as np
from core.knowledge_graph import KnowledgeGraph
from core.link_predict_utils import *
from core.factorizations import *

dataset = 'data/kin'
test_percent = 0.1
corrupt_size = 100
rank = 100


def run_rescal():
    database = KnowledgeGraph()
    database.read_data_from_txt(dataset)
    database.spilt_train_valid_test(0, test_percent)
    num_entities = database.number_of_entities()
    num_relations = database.number_of_relations()

    train_set = database.get_train_set()
    matrix_list = make_sparse_matrix_for_rescal(train_set, num_entities, num_relations)
    a_matrix, r_tensor = rescal(matrix_list, rank)

    test_batch, labels = make_corrupt_for_eval(database.get_test_set(), database,
                                               num_entities, corrupt_size)
    predicts = []
    for triple in test_batch:
        predicts.append(rescal_eval(a_matrix, r_tensor, triple))

    mrr, hit_at_10, auc_pr, ap, prec = metrics_in_a_batch(np.array(predicts), np.array(labels))
    print("Evaluation-------\nTrain:  mmr: %f, hit@10: %f, auc_pr: %f, ap: %f, prec: %f" %
          (mrr, hit_at_10, auc_pr, ap, prec))


def run_tucker():
    database = KnowledgeGraph()
    database.read_data_from_txt(dataset)
    database.spilt_train_valid_test(0, test_percent)
    num_entities = database.number_of_entities()
    num_relations = database.number_of_relations()

    train_set = database.get_train_set()
    tensor = make_tensor_from_triple_list(train_set, num_entities, num_relations)
    predict_tensor = tucker(tensor, rank)

    test_batch, labels = make_corrupt_for_eval(database.get_test_set(), database,
                                               num_entities, corrupt_size)
    predicts = []
    for triple in test_batch:
        predicts.append(predict_tensor[triple[0], triple[1], triple[2]])

    mrr, hit_at_10, auc_pr, ap, prec = metrics_in_a_batch(np.array(predicts), np.array(labels))
    print("Evaluation-------\nTrain:  mmr: %f, hit@10: %f, auc_pr: %f, ap: %f, prec: %f" %
          (mrr, hit_at_10, auc_pr, ap, prec))


def run_cp():
    database = KnowledgeGraph()
    database.read_data_from_txt(dataset)
    database.spilt_train_valid_test(0, test_percent)
    num_entities = database.number_of_entities()
    num_relations = database.number_of_relations()

    train_set = database.get_train_set()
    tensor = make_tensor_from_triple_list(train_set, num_entities, num_relations)
    predict_tensor = cp(tensor, rank)

    test_batch, labels = make_corrupt_for_eval(database.get_test_set(), database,
                                               num_entities, corrupt_size)
    predicts = []
    for triple in test_batch:
        predicts.append(predict_tensor[triple[0], triple[1], triple[2]])

    mrr, hit_at_10, auc_pr, ap, prec = metrics_in_a_batch(np.array(predicts), np.array(labels))
    print("Evaluation-------\nTrain:  mmr: %f, hit@10: %f, auc_pr: %f, ap: %f, prec: %f" %
          (mrr, hit_at_10, auc_pr, ap, prec))


if __name__ == '__main__':
    run_rescal()
    # run_tucker()
    # run_cp()