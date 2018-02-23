import tensorflow as tf

__all__ = ['build_graph']


def build_graph(batch, labels, num_entities, num_relations,
                rank_e, rank_r, num_slice, lambda_para):

    with tf.name_scope('Embeddings'):
        embeddings_e = tf.Variable(
                tf.nn.l2_normalize(tf.truncated_normal(shape=[num_entities, rank_e]), 1),
                name='Embedding_e_matrix')

        embeddings_r = tf.Variable(
                tf.nn.l2_normalize(tf.truncated_normal(shape=[num_relations, rank_r]), 1),
                name='Embedding_r_matrix')

        embeddings_e_trans = tf.transpose(embeddings_e)
        embeddings_r_trans = tf.transpose(embeddings_r)
        embedding_normalize = tf.group(tf.assign(embeddings_e, tf.nn.l2_normalize(embeddings_e, 1)),
                                       tf.assign(embeddings_r, tf.nn.l2_normalize(embeddings_r, 1)))

    with tf.name_scope('Parameters'):
        w_para = tf.Variable(tf.truncated_normal(shape=[num_slice, rank_e * 2 + rank_r]), name='W')
        b_para = tf.Variable(tf.zeros(shape=[num_slice, 1]))
        u_para = tf.Variable(tf.ones(shape=[1, num_slice]), name='u')

    with tf.name_scope('Input'):
        e1_id, e2_id, r_id = tf.split(batch, 3, axis=1)
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
        main_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=score),
                                   name='main_loss')
        weight_decay = sum([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        loss = main_loss + lambda_para * weight_decay

    with tf.name_scope('Optimizer'):
        optimizer = tf.train.AdamOptimizer().minimize(loss)

    with tf.name_scope('Summary/'):
        tf.summary.scalar('Loss', loss)
        tf.summary.histogram('Embedding_entities', embeddings_e)
        tf.summary.histogram('Embedding_relations', embeddings_r)
        merged_summary = tf.summary.merge_all()

    return (predicts, embedding_normalize, optimizer,
            loss, embeddings_e, embeddings_r, merged_summary)
