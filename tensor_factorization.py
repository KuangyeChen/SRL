#!/usr/bin/env python

import numpy as np
from core.knowledge_graph import KnowledgeGraph
from core.link_predict_utils import *
from core.factorizations import *

dataset = 'data/kin_nominal'
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

    test_batch, labels = make_corrupt(database.get_test_set(), database,
                                      num_entities, corrupt_size)
    predicts = []
    for triple in test_batch:
        predicts.append(rescal_eval(a_matrix, r_tensor, triple))

    train_predicts = []
    for triple in train_set:
        train_predicts.append(rescal_eval(a_matrix, r_tensor, triple))
    # print(train_predicts)
    # print(np.percentile(train_predicts, 20))
    # print(predicts)
    # print(labels)
    report_metrics(metrics_in_a_batch(np.array(predicts), np.array(labels)))


def run_tucker():
    database = KnowledgeGraph()
    database.read_data_from_txt(dataset)
    database.spilt_train_valid_test(0, test_percent)
    num_entities = database.number_of_entities()
    num_relations = database.number_of_relations()

    train_set = database.get_train_set()
    tensor = make_tensor_from_triple_list(train_set, num_entities, num_relations)
    predict_tensor = tucker(tensor, rank)

    test_batch, labels = make_corrupt(database.get_test_set(), database,
                                      num_entities, corrupt_size)
    predicts = []
    for triple in test_batch:
        predicts.append(predict_tensor[triple[0], triple[1], triple[2]])

    report_metrics(metrics_in_a_batch(np.array(predicts), np.array(labels)))


def run_cp():
    database = KnowledgeGraph()
    database.read_data_from_txt(dataset)
    database.spilt_train_valid_test(0, test_percent)
    num_entities = database.number_of_entities()
    num_relations = database.number_of_relations()

    train_set = database.get_train_set()
    tensor = make_tensor_from_triple_list(train_set, num_entities, num_relations)
    predict_tensor = cp(tensor, rank)

    test_batch, labels = make_corrupt(database.get_test_set(), database,
                                      num_entities, corrupt_size)
    predicts = []
    for triple in test_batch:
        predicts.append(predict_tensor[triple[0], triple[1], triple[2]])

    report_metrics(metrics_in_a_batch(np.array(predicts), np.array(labels)))


if __name__ == '__main__':
    run_rescal()
    # run_tucker()
    # run_cp()
