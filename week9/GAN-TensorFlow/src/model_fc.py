import tensorflow as tf


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    # return tf.random_normal(shape=size, stddev=xavier_stddev)
    return xavier_stddev


def discriminator(X, reuse=False):
    with tf.variable_scope('discriminator'):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        J = 784
        K = 128
        L = 1

        W1 = tf.get_variable('D_W1', [J, K],
                             initializer=tf.random_normal_initializer(stddev=xavier_init([J, K])))
        B1 = tf.get_variable('D_B1', [K], initializer=tf.constant_initializer())
        W2 = tf.get_variable('D_W2', [K, L],
                             initializer=tf.random_normal_initializer(stddev=xavier_init([K, L])))
        B2 = tf.get_variable('D_B2', [L], initializer=tf.constant_initializer())

        # summary
        tf.summary.histogram('weight1', W1)
        tf.summary.histogram('weight2', W2)
        tf.summary.histogram('biases1', B1)
        tf.summary.histogram('biases2', B2)

        fc1 = tf.nn.relu((tf.matmul(X, W1) + B1))
        logits = tf.matmul(fc1, W2) + B2
        prob = tf.nn.sigmoid(logits)
        return prob, logits


def generator(X):
    with tf.variable_scope('generator'):
        K = 128
        L = 784

        W1 = tf.get_variable('G_W1', [100, K],
                             initializer=tf.random_normal_initializer(stddev=xavier_init([100, K])))
        B1 = tf.get_variable('G_B1', [K], initializer=tf.constant_initializer())
        W2 = tf.get_variable('G_W2', [K, L],
                             initializer=tf.random_normal_initializer(stddev=xavier_init([K, L])))
        B2 = tf.get_variable('G_B2', [L], initializer=tf.constant_initializer())

        # summary
        tf.summary.histogram('weight1', W1)
        tf.summary.histogram('weight2', W2)
        tf.summary.histogram('biases1', B1)
        tf.summary.histogram('biases2', B2)

        fc1 = tf.nn.relu((tf.matmul(X, W1) + B1))
        fc2 = tf.matmul(fc1, W2) + B2
        prob = tf.nn.sigmoid(fc2)

        return prob
