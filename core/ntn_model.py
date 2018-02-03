import tensorflow as tf

__all__ = ['build_graph']


def build_graph(batch_placeholders, corrupt_placeholders, relation_r_empty,
                num_entities, num_relations, embedding_size,
                num_slice, lambda_para, learning_rate):
    """
    Build a Neural Tensor Network in the default graph.
    See also: Socher, Richard, et al. "Reasoning with neural tensor networks for knowledge base completion."
              Advances in neural information processing systems. 2013.

    :param batch_placeholders: List of placeholders.
                               Holding triples for corresponding relation r.
    :param corrupt_placeholders: List of placeholders.
                                 Holding corrupted e2 for corresponding item in batch_placeholders.
    :param relation_r_empty: List of placeholders.
                             Indicating if input has any triples involving relation r.
    :param num_entities: Number of entities.
    :param num_relations: Number of relations
    :param embedding_size: Length of embedding vector.
    :param num_slice: Number of slices used in bilinear term.
    :param lambda_para: Coefficient for L2 weight decay.
    :param learning_rate: Learning rate for AdaGrad.
    :return: predicts: Predicted confidence value for each triples.
             embedding_normalize: TensorFlow op for normalizing embedding vectors.
             optimizer: TensorFlow op for AdaGrad optimizing.
    """
    with tf.name_scope('Embed_table'):
        embeddings = tf.Variable(tf.nn.l2_normalize(tf.truncated_normal(shape=[num_entities, embedding_size]), 1),
                                 name='Embedding_matrix')
        embeddings_t = tf.transpose(embeddings)
        embedding_normalize = tf.assign(embeddings, tf.nn.l2_normalize(embeddings, 1))

    with tf.name_scope('Parameters'):
        w_list = [tf.Variable(tf.truncated_normal(shape=[embedding_size, embedding_size, num_slice]),
                              name='W_%d' % r)
                  for r in range(num_relations)]
        v_list = [tf.Variable(tf.truncated_normal(shape=[num_slice, 2*embedding_size]),
                              name='v_%d' % r)
                  for r in range(num_relations)]
        b_list = [tf.Variable(tf.zeros(shape=[num_slice, 1]),
                              name='bias_%d' % r)
                  for r in range(num_relations)]
        u_list = [tf.Variable(tf.ones(shape=[1, num_slice]),
                              name='u_%d' % r)
                  for r in range(num_relations)]

    predicts = []
    predicts_corrupt = []
    for r in range(num_relations):
        with tf.name_scope('ntn_input/'):
            e1_id, real_e2_id = tf.split(batch_placeholders[r], 2, axis=1)
            fake_e2_id = corrupt_placeholders[r]
            e1 = tf.squeeze(tf.gather(embeddings_t, e1_id, axis=1), [2], name='e1')
            real_e2 = tf.squeeze(tf.gather(embeddings_t, real_e2_id, axis=1), [2], name='true_e2')
            fake_e2 = tf.gather(embeddings_t, fake_e2_id, axis=1, name='fake_e2')

        with tf.name_scope('Bilinear_product/'):
            real_bilinear_product = []
            fake_bilinear_product = []
            for slice_i in range(num_slice):
                real_bilinear_product.append(tf.reduce_sum(
                    e1 * tf.matmul(w_list[r][:, :, slice_i], real_e2), axis=0))
                fake_bilinear_product.append(tf.reduce_sum(
                    e1 * tf.matmul(w_list[r][:, :, slice_i], fake_e2), axis=0))
            real_bilinear_product = tf.stack(real_bilinear_product, name='real_bilinear_product')
            fake_bilinear_product = tf.stack(fake_bilinear_product, name='fake_bilinear_product')

        with tf.name_scope('Standard_layer/'):
            real_std_layer = tf.matmul(v_list[r], tf.concat([e1, real_e2], axis=0), name='real_std_layer')
            fake_std_layer = tf.matmul(v_list[r], tf.concat([e1, fake_e2], axis=0), name='fake_std_layer')
        with tf.name_scope('Preactivation/'):
            real_preactivation = real_bilinear_product + real_std_layer + b_list[r]
            fake_preactivation = fake_bilinear_product + fake_std_layer + b_list[r]

        with tf.name_scope('Activation/'):
            real_activation = tf.tanh(real_preactivation, name='real_activation')
            fake_activation = tf.tanh(fake_preactivation, name='fake_activation')
        with tf.name_scope('Accumulation/'):
            # If relation r is actually not involved in this batch, store no results for relation r.
            real_score = tf.cond(relation_r_empty[r],
                                 lambda: tf.constant([]),
                                 lambda: tf.reshape(tf.matmul(u_list[r], real_activation), shape=[-1]),
                                 name='real_score')
            fake_score = tf.cond(relation_r_empty[r],
                                 lambda: tf.constant([]),
                                 lambda: tf.reshape(tf.matmul(u_list[r], fake_activation), shape=[-1]),
                                 name='fake_score')

        predicts.append(real_score)
        predicts_corrupt.append(fake_score)

    with tf.name_scope('Predicts'):
        predicts = tf.concat(predicts, axis=0, name='predicts')
        predicts_corrupt = tf.concat(predicts_corrupt, axis=0, name='predicts_corrupt')

    with tf.name_scope('Loss_function'):
        main_loss = tf.reduce_sum(tf.maximum(predicts_corrupt - predicts + 1, 0), name='main_loss')
        weight_decay = sum([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        loss = main_loss + lambda_para * weight_decay
    with tf.name_scope('Optimizer'):
        optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

    return predicts, embedding_normalize, optimizer, embeddings
