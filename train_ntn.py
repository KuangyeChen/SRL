#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import core.ntn_model as ntn_model
from core.knowledge_graph import KnowledgeGraph
from core.link_predict_utils import *

dataset = 'data/kin'
num_iter = 300
slice_size = 2
embed_size = 100
corrupt_size_train = 10
corrupt_size_eval = 100
batch_size = 6000
valid_percent = 0.05
test_percent = 0.05
lambda_para = 0.0001
save_per_iter = 100
report_per_iter = 10
learning_rate = 0.1


def fill_feed_dict(placeholders, data_lists, num_relations):
    feed_dict = {}
    for r in range(num_relations):
        for (var, value) in zip(placeholders, data_lists):
            feed_dict[var[r]] = value[r]
    return feed_dict


def run_training():
    database = KnowledgeGraph()
    database.read_data_from_txt(dataset)
    database.spilt_train_valid_test(valid_percent, test_percent)
    num_entities = database.number_of_entities()
    num_relations = database.number_of_relations()

    with tf.Graph().as_default():
        with tf.name_scope('Feed_in'):
            batch_placeholders = [tf.placeholder(shape=[None, 2], name='batch_%d' % r, dtype=tf.int32)
                                  for r in range(num_relations)]
            corrupt_placeholders = [tf.placeholder(shape=[None], name='corrupt_%d' % r, dtype=tf.int32)
                                    for r in range(num_relations)]
            relation_r_empty = [tf.placeholder(shape=[], name='empty_%d' % r, dtype=tf.bool)
                                for r in range(num_relations)]
        print('Building Graph...')
        predicts, embed_normalize, optimizer, embeddings = ntn_model.build_graph(
            batch_placeholders, corrupt_placeholders, relation_r_empty, num_entities,
            num_relations, embed_size, slice_size, lambda_para, learning_rate)

        saver = tf.train.Saver(tf.trainable_variables())
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        print("Start training")
        for step in range(1, num_iter+1):
            print('Iter No.%d' % step)
            print('Get batch')
            batch, corrupt = make_corrupt_for_train(database.get_train_batch(batch_size), database,
                                                    num_entities, corrupt_size_train)
            batch_list, corrupt_list, empty_r = make_split(batch, corrupt, num_relations,
                                                           make_zero_in_extra_list=True)
            feed_dict = fill_feed_dict([batch_placeholders, corrupt_placeholders, relation_r_empty],
                                       [batch_list, corrupt_list, empty_r], num_relations)
            print('Training...')
            sess.run(optimizer, feed_dict=feed_dict)
            sess.run(embed_normalize)

            if step % report_per_iter == 0:
                print('Evaluating...')

                # Evaluation on one batch of train set.
                valid_set, labels = make_corrupt_for_eval(database.get_train_batch(batch_size), database,
                                                          num_entities, corrupt_size_eval)
                valid_list, label_list, empty_r = make_split(valid_set, labels, num_relations)
                labels = np.hstack(label_list)

                feed_dict = fill_feed_dict([batch_placeholders, relation_r_empty],
                                           [valid_list, empty_r], num_relations)
                valid_predicts = sess.run(predicts, feed_dict=feed_dict)

                mrr, hit_at_10, auc_pr, ap, prec = metrics_in_a_batch(valid_predicts, labels)
                print("Train batch evaluation:  mmr: %f, hit@10: %f, auc_pr: %f, ap: %f, prec: %f" %
                      (mrr, hit_at_10, auc_pr, ap, prec))

                # Evaluation on validation set.
                valid_set, labels = make_corrupt_for_eval(database.get_valid_set(), database,
                                                          num_entities, corrupt_size_eval)
                valid_list, label_list, empty_r = make_split(valid_set, labels, num_relations)
                labels = np.hstack(label_list)

                feed_dict = fill_feed_dict([batch_placeholders, relation_r_empty],
                                           [valid_list, empty_r], num_relations)
                valid_predicts = sess.run(predicts, feed_dict=feed_dict)
                mrr, hit_at_10, auc_pr, ap, prec = metrics_in_a_batch(valid_predicts, labels)
                print("Validation  evaluation:  mmr: %f, hit@10: %f, auc_pr: %f, ap: %f, prec: %f" %
                      (mrr, hit_at_10, auc_pr, ap, prec))

            if step % save_per_iter == 0:
                print('Saving model')
                saver.save(sess, './tmp/saved_model', global_step=step)

        # Evaluation on test set.
        valid_set, labels = make_corrupt_for_eval(database.get_test_set(), database,
                                                  num_entities, corrupt_size_eval)
        valid_list, label_list, empty_r = make_split(valid_set, labels, num_relations)
        labels = np.hstack(label_list)

        feed_dict = fill_feed_dict([batch_placeholders, relation_r_empty],
                                   [valid_list, empty_r], num_relations)
        test_predicts = sess.run(predicts, feed_dict)

        mrr, hit_at_10, auc_pr, ap, prec = metrics_in_a_batch(test_predicts, labels)
        print("Final evaluation-------\nTrain:  mmr: %f, hit@10: %f, auc_pr: %f, ap: %f, prec: %f" %
              (mrr, hit_at_10, auc_pr, ap, prec))


if __name__ == "__main__":
    run_training()
