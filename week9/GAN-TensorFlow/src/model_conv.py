import tensorflow as tf


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    # return tf.random_normal(shape=size, stddev=xavier_stddev)
    return xavier_stddev


def conv(x, w, b, stride, name):
    with tf.variable_scope('conv'):
        tf.summary.histogram('weight', w)
        tf.summary.histogram('biases', b)
        return tf.nn.conv2d(x,
                           filter=w,
                           strides=[1, stride, stride, 1],
                           padding='SAME',
                           name=name) + b


def deconv(x, w, b, shape, stride, name):
    with tf.variable_scope('deconv'):
        tf.summary.histogram('weight', w)
        tf.summary.histogram('biases', b)
        return tf.nn.conv2d_transpose(x,
                                       filter=w,
                                       output_shape=shape,
                                       strides=[1, stride, stride, 1],
                                       padding='SAME',
                                       name=name) + b


def lrelu(x, alpha=0.2):
    with tf.variable_scope('leakyReLU'):
        return tf.maximum(x, alpha * x)


def discriminator(X, reuse=False):
    with tf.variable_scope('discriminator'):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        K = 64
        M = 128
        N = 256

        W1 = tf.get_variable('D_W1', [4, 4, 1, K], initializer=tf.random_normal_initializer(stddev=0.1))
        B1 = tf.get_variable('D_B1', [K], initializer=tf.constant_initializer())
        W2 = tf.get_variable('D_W2', [4, 4, K, M], initializer=tf.random_normal_initializer(stddev=0.1))
        B2 = tf.get_variable('D_B2', [M], initializer=tf.constant_initializer())
        W3 = tf.get_variable('D_W3', [7*7*M, N], initializer=tf.random_normal_initializer(stddev=0.1))
        B3 = tf.get_variable('D_B3', [N], initializer=tf.constant_initializer())
        W4 = tf.get_variable('D_W4', [N, 1], initializer=tf.random_normal_initializer(stddev=0.1))
        B4 = tf.get_variable('D_B4', [1], initializer=tf.constant_initializer())

        X = tf.reshape(X, [-1, 28, 28, 1], 'reshape')

        conv1 = conv(X, W1, B1, stride=2, name='conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1)
        conv2 = conv(tf.nn.dropout(lrelu(bn1), 0.4), W2, B2, stride=2, name='conv2')
        # conv2 = conv(lrelu(conv1), W2, B2, stride=2, name='conv2')

        bn2 = tf.contrib.layers.batch_norm(conv2)
        flat = tf.reshape(tf.nn.dropout(lrelu(bn2), 0.4), [-1, 7*7*M], name='flat')
        # flat = tf.reshape(lrelu(conv2), [-1, 7*7*M], name='flat')

        dense = lrelu(tf.matmul(flat, W3) + B3)
        logits = tf.matmul(dense, W4) + B4
        prob = tf.nn.sigmoid(logits)
        return prob, logits


def generator(X, batch_size=64):
    with tf.variable_scope('generator'):

        K = 256
        L = 128
        M = 64

        W1 = tf.get_variable('G_W1', [100, 7*7*K], initializer=tf.random_normal_initializer(stddev=0.1))
        B1 = tf.get_variable('G_B1', [7*7*K], initializer=tf.constant_initializer())

        W2 = tf.get_variable('G_W2', [4, 4, M, K], initializer=tf.random_normal_initializer(stddev=0.1))
        B2 = tf.get_variable('G_B2', [M], initializer=tf.constant_initializer())

        W3 = tf.get_variable('G_W3', [4, 4, 1, M], initializer=tf.random_normal_initializer(stddev=0.1))
        B3 = tf.get_variable('G_B3', [1], initializer=tf.constant_initializer())

        X = lrelu(tf.matmul(X, W1) + B1)
        X = tf.reshape(X, [batch_size, 7, 7, K])
        deconv1 = deconv(X, W2, B2, shape=[batch_size, 14, 14, M], stride=2, name='deconv1')
        bn1 = tf.contrib.layers.batch_norm(deconv1)
        deconv2 = deconv(tf.nn.dropout(lrelu(bn1), 0.4), W3, B3, shape=[batch_size, 28, 28, 1], stride=2, name='deconv2')

        XX = tf.reshape(deconv2, [-1, 28*28], 'reshape')

        return tf.nn.sigmoid(XX)

