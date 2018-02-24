import tensorflow as tf
import collections
from core.link_predict_utils import metrics_in_a_batch, report_metrics

__all__ = ['report_log_metrics',
           'ntn_model',
           'er_mlp_model']


def report_log_metrics(predicts, labels, logger, step):
    """
    Print and log metrics with TensorFlow summary.

    :param predicts: Predictions in probability.
    :param labels: True labels.
    :param logger: TensorFlow's FileWriter.
    :param step: Global step at training.
    :return: None
    """

    metrics = metrics_in_a_batch(predicts, labels)
    logger.add_summary(tf.Summary(
        value=[tf.Summary.Value(tag='Summary/MRR', simple_value=metrics.mrr),
               tf.Summary.Value(tag='Summary/HIT@10', simple_value=metrics.hit_at_10),
               tf.Summary.Value(tag='Summary/HIT@5', simple_value=metrics.hit_at_5),
               tf.Summary.Value(tag='Summary/HIT@1', simple_value=metrics.hit_at_1),
               tf.Summary.Value(tag='Summary/AUC_PR', simple_value=metrics.auc_pr),
               tf.Summary.Value(tag='Summary/Average_precision', simple_value=metrics.ap),
               tf.Summary.Value(tag='Summary/Precision@0.5', simple_value=metrics.precision)]), step)
    report_metrics(metrics)


def ntn_model(num_entities, num_relations, rank, num_slice, lambda_para):
    """
    Build a Neural Tensor Network in the default graph.
    See also: Socher, Richard, et al. "Reasoning with neural tensor networks for knowledge base completion."
              Advances in neural information processing systems. 2013.

    :param num_entities: Number of entities.
    :param num_relations: Number of relations
    :param rank: Length of embedding vector.
    :param num_slice: Number of slices used in bilinear term.
    :param lambda_para: Coefficient for L2 weight decay.
    :return: A handle containing references for training and evaluation.
    """

    with tf.name_scope('Placeholders/'):
        batch_input = [tf.placeholder(shape=[None, 2], name='batch_%d' % r, dtype=tf.int32)
                       for r in range(num_relations)]
        r_empty_input = [tf.placeholder(shape=[], name='empty_%d' % r, dtype=tf.bool)
                         for r in range(num_relations)]
        labels_input = tf.placeholder(shape=[None], name='labels', dtype=tf.float32)

    with tf.name_scope('Embeddings/'):
        embeddings = tf.Variable(tf.nn.l2_normalize(tf.truncated_normal(shape=[num_entities, rank]), 1),
                                 name='Embedding_matrix')
        embeddings_t = tf.transpose(embeddings)

    with tf.name_scope('Parameters/'):
        w_list = [tf.Variable(tf.truncated_normal(shape=[rank, rank, num_slice]),
                              name='W_%d' % r)
                  for r in range(num_relations)]
        v_list = [tf.Variable(tf.truncated_normal(shape=[num_slice, 2 * rank]),
                              name='v_%d' % r)
                  for r in range(num_relations)]
        b_list = [tf.Variable(tf.zeros(shape=[num_slice, 1]),
                              name='bias_%d' % r)
                  for r in range(num_relations)]
        u_list = [tf.Variable(tf.ones(shape=[1, num_slice]),
                              name='u_%d' % r)
                  for r in range(num_relations)]

    scores = []
    for r in range(num_relations):
        with tf.name_scope('Input/'):
            e1_id, e2_id = tf.split(batch_input[r], 2, axis=1)
            e1 = tf.squeeze(tf.gather(embeddings_t, e1_id, axis=1), [2], name='e1')
            e2 = tf.squeeze(tf.gather(embeddings_t, e2_id, axis=1), [2], name='true_e2')

        with tf.name_scope('Bilinear_product/'):
            bilinear_product = []
            for slice_i in range(num_slice):
                bilinear_product.append(tf.reduce_sum(
                    e1 * tf.matmul(w_list[r][:, :, slice_i], e2), axis=0))
            bilinear_product = tf.stack(bilinear_product, name='bilinear_product')

        with tf.name_scope('Standard_layer/'):
            std_layer = tf.matmul(v_list[r], tf.concat([e1, e2], axis=0), name='std_layer')
        with tf.name_scope('Preactivation/'):
            preactivation = bilinear_product + std_layer + b_list[r]

        with tf.name_scope('Activation/'):
            activation = tf.tanh(preactivation, name='activation')
        with tf.name_scope('Accumulation/'):
            # If relation r is actually not involved in this batch, store no results for relation r.
            score = tf.cond(r_empty_input[r],
                            lambda: tf.constant([]),
                            lambda: tf.reshape(tf.matmul(u_list[r], activation), shape=[-1]))
        scores.append(score)

    with tf.name_scope('Output/'):
        score = tf.concat(scores, axis=0, name='score')
        predicts = tf.sigmoid(score, name='predicts')

    with tf.name_scope('Loss_function/'):
        main_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_input, logits=score),
                                   name='main_loss')
        weight_decay = sum([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        loss = main_loss + lambda_para * weight_decay

    with tf.name_scope('Optimizer/'):
        optimizer = tf.train.AdamOptimizer().minimize(loss)
        with tf.control_dependencies([optimizer]):
            optimize_normalize = tf.assign(embeddings, tf.nn.l2_normalize(embeddings, 1))

    with tf.name_scope('Summary/'):
        tf.summary.scalar('Loss', loss)
        tf.summary.histogram('Embeddings', embeddings)
        merged_summary = tf.summary.merge_all()

    handle = collections.namedtuple('handle', ['batch_input',
                                               'labels_input',
                                               'r_empty_input',
                                               'predicts',
                                               'loss',
                                               'optimize',
                                               'embeddings',
                                               'summary'])

    return handle(batch_input, labels_input, r_empty_input, predicts,
                  loss, optimize_normalize, embeddings, merged_summary)


def er_mlp_model(num_entities, num_relations, rank_e, rank_r, num_slice, lambda_para):
    """
    Build a ER_MLP in default graph.

    :param num_entities: Number of entities
    :param num_relations: Number of relations
    :param rank_e: Length of embedding vector for entities.
    :param rank_r: Length of embedding vector for relations.
    :param num_slice: Size of the hidden layer.
    :param lambda_para: Coefficient for L2 weight decay.
    :return: A handle containing references for training and evaluation.
    """

    with tf.name_scope('Placeholders'):
        batch_input = tf.placeholder(shape=[None, 3], name='batch', dtype=tf.int32)
        labels_input = tf.placeholder(shape=[None], name='labels', dtype=tf.float32)

    with tf.name_scope('Embeddings'):
        embeddings_e = tf.Variable(
                tf.nn.l2_normalize(tf.truncated_normal(shape=[num_entities, rank_e]), 1),
                name='Embedding_e_matrix')

        embeddings_r = tf.Variable(
                tf.nn.l2_normalize(tf.truncated_normal(shape=[num_relations, rank_r]), 1),
                name='Embedding_r_matrix')

        embeddings_e_trans = tf.transpose(embeddings_e)
        embeddings_r_trans = tf.transpose(embeddings_r)

    with tf.name_scope('Parameters'):
        w_para = tf.Variable(tf.truncated_normal(shape=[num_slice, rank_e * 2 + rank_r]), name='W')
        b_para = tf.Variable(tf.zeros(shape=[num_slice, 1]))
        u_para = tf.Variable(tf.ones(shape=[1, num_slice]), name='u')

    with tf.name_scope('Input'):
        e1_id, e2_id, r_id = tf.split(batch_input, 3, axis=1)
        e1 = tf.squeeze(tf.gather(embeddings_e_trans, e1_id, axis=1), [2], name='e1')
        e2 = tf.squeeze(tf.gather(embeddings_e_trans, e2_id, axis=1), [2], name='e2')
        r = tf.squeeze(tf.gather(embeddings_r_trans, r_id, axis=1), [2], name='r')

        er_input = tf.concat([e1, e2, r], axis=0, name='er_input')

    with tf.name_scope('Fully_connected_layer'):
        preactivation = tf.matmul(w_para, er_input) + b_para
        activation = tf.tanh(preactivation, name='activation')

    with tf.name_scope('Output'):
        score = tf.reshape(tf.matmul(u_para, activation), shape=[-1])
        predicts = tf.sigmoid(score, name='predicts')

    with tf.name_scope('Loss_function'):
        main_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_input, logits=score),
                                   name='main_loss')
        weight_decay = sum([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        loss = main_loss + lambda_para * weight_decay

    with tf.name_scope('Optimizer'):
        optimizer = tf.train.AdamOptimizer().minimize(loss)
        with tf.control_dependencies([optimizer]):
            optimize_normalize = tf.group(tf.assign(embeddings_e, tf.nn.l2_normalize(embeddings_e, 1)),
                                          tf.assign(embeddings_r, tf.nn.l2_normalize(embeddings_r, 1)))

    with tf.name_scope('Summary/'):
        tf.summary.scalar('Loss', loss)
        tf.summary.histogram('Embedding_entities', embeddings_e)
        tf.summary.histogram('Embedding_relations', embeddings_r)
        merged_summary = tf.summary.merge_all()

    handle = collections.namedtuple('handle', ['batch_input',
                                               'labels_input',
                                               'predicts',
                                               'loss',
                                               'optimize',
                                               'embeddings_e',
                                               'embeddings_r',
                                               'summary'])

    return handle(batch_input, labels_input, predicts, loss, optimize_normalize,
                  embeddings_e, embeddings_r, merged_summary)
