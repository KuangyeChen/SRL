#!/usr/bin/env python

import tensorflow as tf
import shutil
import os
from core.neural_networks import ntn_model, report_log_metrics
from core.knowledge_graph import KnowledgeGraph
from core.link_predict_utils import *

dataset = 'data/kin_nominal'
# Hyperparameters
slice_size = 1
rank = 100
valid_percent = 0.05
test_percent = 0.05
lambda_para = 1e-4
pos_weight = 10

# Parameters for training
batch_size = 8000
max_iter = 100000
corrupt_size_train = 10
corrupt_size_eval = 100
save_per_iter = 300
eval_per_iter = 100


def main():
    # Early stop condition for training
    min_loss = float('inf')
    eval_no_improve = 0
    max_eval_no_improve = 10

    # Clean tmp files
    if os.path.exists('./tmp'):
        decide = input('Clean tmp/ folder? (y/n)')
        if decide.lower() == 'y' or decide.lower() == 'yes':
            shutil.rmtree('./tmp/')
            print('tmp/ cleaned.')

    # Read database
    database = KnowledgeGraph()
    database.read_data_from_txt(dataset)
    database.spilt_train_valid_test(valid_percent, test_percent)

    # Build Graph
    ntn_graph = tf.Graph()
    with ntn_graph.as_default():
        print('Building Graph...')
        handle = ntn_model(database.number_of_entities(), database.number_of_relations(),
                           rank, slice_size, lambda_para, pos_weight)
        print('Graph built.')
        saver = tf.train.Saver(tf.trainable_variables())
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

    # Tensorflow logs
    train_writer = tf.summary.FileWriter('./tmp/log/train', ntn_graph)
    valid_writer = tf.summary.FileWriter('./tmp/log/valid')

    print("Start training")
    for step in range(1, max_iter + 1):
        train_summary, train_loss, _ = run_once(database.get_train_batch(batch_size), database,
                                                sess, handle, corrupt_size_train, if_test=False)
        train_writer.add_summary(train_summary, step)
        print('Step %d: Loss = %f' % (step, train_loss), end='\r')

        if step % eval_per_iter == 0:
            print('--------------- Evaluation --------------')
            print('Training set')
            _, train_valid_loss, train_valid_predicts, train_valid_labels = run_once(
                database.get_train_batch(batch_size), database, sess, handle, corrupt_size_eval)
            report_log_metrics(train_valid_predicts, train_valid_labels, train_writer, step)

            print('Validation set')
            valid_summary, valid_loss, valid_predicts, valid_labels = run_once(
                database.get_valid_set(), database, sess, handle, corrupt_size_eval)
            valid_writer.add_summary(valid_summary, step)
            report_log_metrics(valid_predicts, valid_labels, valid_writer, step)
            print('**Validation loss = %f' % valid_loss)
            print('-----------------------------------------')

            # Stop training when validation loss stop decreasing.
            if valid_loss < min_loss:
                min_loss = valid_loss
                eval_no_improve = 0
            else:
                eval_no_improve = eval_no_improve + 1

            if eval_no_improve >= max_eval_no_improve:
                break

        if step % save_per_iter == 0:
            saver.save(sess, './tmp/save/ntn_model', global_step=step)
            print('**** Saved model at step %d ****' % step)
    print('Training done.')

    # Evaluation on test set.
    print('############# Test Evaluation #############')
    print('++++ Checkpoints status ++++')
    checkpoint_status = tf.train.get_checkpoint_state('tmp/save/')
    print(checkpoint_status, end='')
    print('++++++++++++++++++++++++++++')

    for checkpoint in checkpoint_status.all_model_checkpoint_paths:
        step = checkpoint.split('-')[-1]
        print('------- Test model in step %s --------' % step)
        saver.restore(sess, checkpoint)
        _, test_loss, test_predicts, test_labels = run_once(
            database.get_test_set(), database, sess, handle, corrupt_size_eval)

        print('Test loss: %f' % test_loss)
        report_metrics(metrics_in_a_batch(test_predicts, test_labels))
    print('--------------------------------------')


def fill_feed_dict(input_list_r, data_list_r, input_list, data_list, num_relations):
    feed_dict = {}
    for (var, value) in zip(input_list_r, data_list_r):
        for r in range(num_relations):
            feed_dict[var[r]] = value[r]

    for (var, value) in zip(input_list, data_list):
        feed_dict[var] = value

    return feed_dict


def run_once(data, database, sess, handle, corrupt_size, if_test=True):
    num_relations = database.number_of_relations()

    batch, labels = make_corrupt(data, database, corrupt_size)
    batch_list, labels, r_empty = make_split(batch, labels, num_relations)
    feed_dict = fill_feed_dict([handle.batch_input, handle.r_empty_input],
                               [batch_list,          r_empty],
                               [handle.labels_input],
                               [labels],
                               num_relations)

    if not if_test:
        return sess.run([handle.summary, handle.loss, handle.optimize], feed_dict=feed_dict)

    summary, loss, predicts = sess.run([handle.summary, handle.loss, handle.predicts], feed_dict=feed_dict)
    return summary, loss, predicts, labels


if __name__ == "__main__":
    main()
