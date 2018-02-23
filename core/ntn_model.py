import tensorflow as tf

__all__ = ['build_graph']


def build_graph(batch, labels, relation_r_empty, num_entities, num_relations,
                embedding_size, num_slice, lambda_para):
    """
    Build a Neural Tensor Network in the default graph.
    See also: Socher, Richard, et al. "Reasoning with neural tensor networks for knowledge base completion."
              Advances in neural information processing systems. 2013.

    :param batch: List of placeholders.
                  Holding triples for corresponding relation r.
    :param labels: 1D array of labels.
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

    scores = []
    for r in range(num_relations):
        with tf.name_scope('ntn_input/'):
            e1_id, e2_id = tf.split(batch[r], 2, axis=1)
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
            score = tf.cond(relation_r_empty[r],
                            lambda: tf.constant([]),
                            lambda: tf.reshape(tf.matmul(u_list[r], activation), shape=[-1]))
        scores.append(score)

    with tf.name_scope('Predicts'):
        score = tf.concat(scores, axis=0, name='score')
        predicts = tf.sigmoid(score, name='predicts')

    with tf.name_scope('Loss_function'):
        main_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=score),
                                   name='main_loss')
        weight_decay = sum([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        loss = main_loss + lambda_para * weight_decay
        mean_loss = main_loss / tf.cast(tf.shape(labels)[0], tf.float32) + lambda_para * weight_decay

    with tf.name_scope('Optimizer'):
        optimizer = tf.train.AdamOptimizer().minimize(loss)

    with tf.name_scope('Summary/'):
        tf.summary.scalar('Loss', mean_loss)
        tf.summary.histogram('Embeddings', embeddings)
        merged_summary = tf.summary.merge_all()

    return predicts, embedding_normalize, optimizer, mean_loss, embeddings, merged_summary
