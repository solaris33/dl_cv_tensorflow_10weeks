import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import argparse


def read_data():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
    return mnist


def plot(samples):
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
    return fig


def train(logdir, batch_size):
    from model_conv import discriminator, generator

    mnist = read_data()

    with tf.variable_scope('placeholder'):
        # Raw image
        X = tf.placeholder(tf.float32, [None, 784])
        tf.summary.image('raw image', tf.reshape(X, [-1, 28, 28, 1]), 3)
        # Noise
        z = tf.placeholder(tf.float32, [None, 100])  # noise
        tf.summary.histogram('Noise', z)

    with tf.variable_scope('GAN'):
        G = generator(z, batch_size)

        D_real, D_real_logits = discriminator(X, reuse=False)
        D_fake, D_fake_logits = discriminator(G, reuse=True)
    tf.summary.image('generated image', tf.reshape(G, [-1, 28, 28, 1]), 3)

    with tf.variable_scope('Prediction'):
        tf.summary.histogram('real', D_real)
        tf.summary.histogram('fake', D_fake)

    with tf.variable_scope('D_loss'):
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=D_real_logits, labels=tf.ones_like(D_real_logits)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=D_fake_logits, labels=tf.zeros_like(D_fake_logits)))
        d_loss = d_loss_real + d_loss_fake

        tf.summary.scalar('d_loss_real', d_loss_real)
        tf.summary.scalar('d_loss_fake', d_loss_fake)
        tf.summary.scalar('d_loss', d_loss)

    with tf.name_scope('G_loss'):
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
                                (logits=D_fake_logits, labels=tf.ones_like(D_fake_logits)))
        tf.summary.scalar('g_loss', g_loss)

    tvar = tf.trainable_variables()
    dvar = [var for var in tvar if 'discriminator' in var.name]
    gvar = [var for var in tvar if 'generator' in var.name]

    with tf.name_scope('train'):
        d_train_step = tf.train.AdamOptimizer().minimize(d_loss, var_list=dvar)
        g_train_step = tf.train.AdamOptimizer().minimize(g_loss, var_list=gvar)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter('tmp/'+'gan_conv_'+logdir)
    writer.add_graph(sess.graph)

    num_img = 0
    if not os.path.exists('output/'):
        os.makedirs('output/')

    for i in range(100000):
        batch_X, _ = mnist.train.next_batch(batch_size)
        batch_noise = np.random.uniform(-1., 1., [batch_size, 100])

        if i % 500 == 0:
            samples = sess.run(G, feed_dict={z: np.random.uniform(-1., 1., [64, 100])})
            fig = plot(samples)
            plt.savefig('output/%s.png' % str(num_img).zfill(3), bbox_inches='tight')
            num_img += 1
            plt.close(fig)

        _, d_loss_print = sess.run([d_train_step, d_loss],
                                   feed_dict={X: batch_X, z: batch_noise})

        _, g_loss_print = sess.run([g_train_step, g_loss],
                                   feed_dict={z: batch_noise})

        if i % 100 == 0:
            s = sess.run(merged_summary, feed_dict={X: batch_X, z: batch_noise})
            writer.add_summary(s, i)
            print('epoch:%d g_loss:%f d_loss:%f' % (i, g_loss_print, d_loss_print))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train vanila GAN using convolutional networks')
    parser.add_argument('--logdir', type=str, default='1', help='logdir for Tensorboard, give a string')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size: give a int')
    args = parser.parse_args()

    train(logdir=args.logdir, batch_size=args.batch_size)
