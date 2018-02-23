#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import core.ntn_model as ntn_model
from core.knowledge_graph import KnowledgeGraph
from core.link_predict_utils import *

dataset = 'data/kin_nominal'
num_iter = 300
slice_size = 2
rank = 100
corrupt_size_train = 10
corrupt_size_eval = 3
batch_size = 6000
valid_percent = 0.05
test_percent = 0.05
lambda_para = 0.0001
save_per_iter = 100
report_per_iter = 10
learning_rate = 0.1


def fill_feed_dict(input_list_r, data_list_r, input_list, data_list, num_relations):
    feed_dict = {}
    for (var, value) in zip(input_list_r, data_list_r):
        for r in range(num_relations):
            feed_dict[var[r]] = value[r]

    for (var, value) in zip(input_list, data_list):
        feed_dict[var] = value

    return feed_dict


def ntn_evaluation(batch, labels, num_relations, sess, predicts,
                   batch_input, labels_input, r_empty_input):

    batch_list, labels, r_empty = make_split(batch, labels, num_relations)
    feed_dict = fill_feed_dict([batch_input, r_empty_input], [batch_list, r_empty],
                               [labels_input], [labels], num_relations)

    valid_predicts = sess.run(predicts, feed_dict=feed_dict)

    mrr, hit_at_10, auc_pr, ap, precision, num_pos = metrics_in_a_batch(valid_predicts, labels)
    print("mrr: %f, hit@10: %f, auc_pr: %f, ap: %f, positive predicts: %d, precision: %f" %
          (mrr, hit_at_10, auc_pr, ap, num_pos, precision))


def run_training():
    database = KnowledgeGraph()
    database.read_data_from_txt(dataset)
    database.spilt_train_valid_test(valid_percent, test_percent)
    num_entities = database.number_of_entities()
    num_relations = database.number_of_relations()

    with tf.Graph().as_default():
        with tf.name_scope('Feed_in'):
            batch_input = [tf.placeholder(shape=[None, 2], name='batch_%d' % r, dtype=tf.int32)
                           for r in range(num_relations)]
            r_empty_input = [tf.placeholder(shape=[], name='empty_%d' % r, dtype=tf.bool)
                             for r in range(num_relations)]
            labels_input = tf.placeholder(shape=[None], name='corrupt', dtype=tf.float32)
        print('Building Graph...')
        predicts, embed_normalize, optimizer, loss, _, summary = ntn_model.build_graph(
            batch_input, labels_input, r_empty_input, num_entities,
            num_relations, rank, slice_size, lambda_para, learning_rate)

        saver = tf.train.Saver(tf.trainable_variables())
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        print("Start training")
        for step in range(1, num_iter+1):
            print('Iter No.%d' % step)
            print('Get batch')
            batch, labels = make_corrupt(database.get_train_batch(batch_size), database,
                                         num_entities, corrupt_size_train)
            batch_list, labels, r_empty = make_split(batch, labels, num_relations)
            labels = np.hstack(labels)
            feed_dict = fill_feed_dict([batch_input, r_empty_input], [batch_list, r_empty],
                                       [labels_input], [labels], num_relations)
            print('Training...')
            _, l = sess.run([optimizer, loss], feed_dict=feed_dict)
            sess.run(embed_normalize)
            print('loss ', l)

            if step % report_per_iter == 0:
                print('Evaluating on training set...')
                train_valid_set, train_valid_labels = make_corrupt(database.get_train_batch(batch_size), database,
                                                                   num_entities, corrupt_size_eval)
                ntn_evaluation(train_valid_set, train_valid_labels, num_relations, sess, predicts,
                               batch_input, labels_input, r_empty_input)

                print('Evaluating on validation set...')
                valid_set, valid_labels = make_corrupt(database.get_valid_set(), database,
                                                       num_entities, corrupt_size_eval)
                ntn_evaluation(valid_set, valid_labels, num_relations, sess, predicts,
                               batch_input, labels_input, r_empty_input)

            if step % save_per_iter == 0:
                print('Saving model')
                saver.save(sess, './tmp/saved_model', global_step=step)

        # Evaluation on test set.
        print('Final test set evaluation--------------------------')
        test_set, test_labels = make_corrupt(database.get_test_set(), database,
                                             num_entities, corrupt_size_eval)
        ntn_evaluation(test_set, test_labels, num_relations, sess, predicts,
                       batch_input, labels_input, r_empty_input)


if __name__ == "__main__":
    run_training()
