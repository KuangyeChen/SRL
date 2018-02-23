#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import core.er_mlp_model as er_mlp_model
from core.knowledge_graph import KnowledgeGraph
from core.link_predict_utils import *

dataset = 'data/kin'
num_iter = 10000
num_slice = 100
rank_e = 100
rank_r = 100
corrupt_size_train = 10
corrupt_size_eval = 50
batch_size = 6000
valid_percent = 0.05
test_percent = 0.05
lambda_para = 0.5
save_per_iter = 1000
report_per_iter = 50


def calc_report_eval(predicts, labels):
    mrr, hit_at_10, auc_pr, ap, precision, num_pos= metrics_in_a_batch(predicts, labels)
    print("Train batch evaluation:  mmr: %f, hit@10: %f, auc_pr: %f, ap: %f, positive predicts: %d, precision: %f" %
          (mrr, hit_at_10, auc_pr, ap, num_pos, precision))


def run_training():
    database = KnowledgeGraph()
    database.read_data_from_txt(dataset)
    database.spilt_train_valid_test(valid_percent, test_percent)
    num_entities = database.number_of_entities()
    num_relations = database.number_of_relations()

    with tf.Graph().as_default():
        with tf.name_scope('Feed_in'):
            batch_input = tf.placeholder(shape=[None, 3], name='batch', dtype=tf.int32)
            labels_input = tf.placeholder(shape=[None], name='labels', dtype=tf.float32)

        print('Building Graph...')
        predicts, embed_normalize, optimizer, loss, _, _, _ = er_mlp_model.build_graph(
            batch_input, labels_input, num_entities, num_relations,
            rank_e, rank_r, num_slice, lambda_para)

        saver = tf.train.Saver(tf.trainable_variables())
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        print("Start training")
        for step in range(1, num_iter+1):
            print('Iter No.%d' % step)
            print('Get batch')
            batch, labels = make_corrupt(database.get_train_batch(batch_size), database,
                                         num_entities, corrupt_size_train)

            print('Training...')
            _, l = sess.run([optimizer, loss], feed_dict={batch_input: batch, labels_input: labels})
            # sess.run(embed_normalize)
            print("loss ", l)
            # print('score ', sbb)
            if step % report_per_iter == 0:
                print('Evaluating on training set...')
                train_valid_set, train_valid_labels = make_corrupt(database.get_train_batch(batch_size), database,
                                                                   num_entities, corrupt_size_eval)
                train_valid_predicts = sess.run(predicts, feed_dict={batch_input: train_valid_set})
                calc_report_eval(train_valid_predicts, np.array(train_valid_labels))

                print('Evaluating on validation set...')
                valid_set, valid_labels = make_corrupt(database.get_valid_set(), database,
                                                       num_entities, corrupt_size_eval)
                valid_predicts = sess.run(predicts, feed_dict={batch_input: valid_set})
                calc_report_eval(valid_predicts, np.array(valid_labels))

            if step % save_per_iter == 0:
                print('Saving model')
                saver.save(sess, './tmp/saved_model', global_step=step)

        print('Final test evaluation--------------------')
        test_set, test_labels = make_corrupt(database.get_test_set(), database,
                                             num_entities, corrupt_size_eval)
        test_predicts = sess.run(predicts, feed_dict={batch_input: test_set})
        calc_report_eval(test_predicts, np.array(test_labels))


if __name__ == "__main__":
    run_training()