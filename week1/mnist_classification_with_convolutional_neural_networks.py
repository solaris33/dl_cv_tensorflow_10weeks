# -*- coding: utf-8 -*-

# Convolutional Neural Networks(CNNs)를 이용한 Deep MNIST 분류기(Classifier)

# 절대 임포트 설정 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 필요한 라이브러리들을 임포트
import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def deepnn(x):
  """숫자를 분류하기 위한 Deep Neural Networks 그래프를 생성한다.
  인자들(Args):
    x: (N_examples, 784) 차원을 가진 input tensor, 784 일반적인 MNIST 데이터의 픽셀 개수이다.
  리턴값들(Returns):
    tuple (y, keep_prob). y는 (N_examples, 10)형태의 숫자(0-9) tensor이다. 
    keep_prob는 dropout을 위한 scalar placeholder이다.
  """
  # Convolutional Neural Netwokrs(CNNs)를 위한 reshape.
  # 마지막 차원(dimension)은 특징들("features")을 나타낸다.-이 코드에서는 이미지가 grayscale이라 일차원이지만, RGB 이미지라면 3차원, RGBA라면 4차원 이미지 일 것이다.
  x_image = tf.reshape(x, [-1, 28, 28, 1])

  # 첫번째 convolutional layer - 하나의 grayscale 이미지를 32개의 특징들(feature)으로 맵핑(maping)한다.
  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - 2X만큼 downsample한다.
  h_pool1 = max_pool_2x2(h_conv1)

  # 두번째 convolutional layer -- 32개의 특징들(feature)을 64개의 특징들(feature)로 맵핑(maping)한다.
  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # 두번째 pooling layer.
  h_pool2 = max_pool_2x2(h_conv2)

  # Fully Connected Layer 1 -- 2번의 downsampling 이후에, 우리의 28x28 이미지는 7x7x64 특징들(feature map)이 된다.
  # 이를 1024개의 특징들로 맵핑(maping)한다.
  W_fc1 = weight_variable([7 * 7 * 64, 1024])
  b_fc1 = bias_variable([1024])

  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - 모델의 복잡도를 컨트롤한다. 특징들의 co-adaptation을 방지한다.
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # 1024개의 특징들(feature)을 10개의 클래스-숫자 0-9-로 맵핑(maping)한다.
  W_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])

  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d는 full stride를 가진 2d convolution layer를 반환(return)한다."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2는 특징들(feature map)을 2X만큼 downsample한다."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable는 주어진 shape에 대한 weight variable을 생성한다."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable 주어진 shape에 대한 bias variable을 생성한다."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def main(_):
  # data를 import한다.
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # 모델을 생성한다.
  x = tf.placeholder(tf.float32, [None, 784])

  # loss와 optimizer를 정의한다.
  y_ = tf.placeholder(tf.float32, [None, 10])

  # Deep Neural Networks 그래프를 생성한다.
  y_conv, keep_prob = deepnn(x)

  # Cross Entropy를 비용함수(loss function)으로 정의하고, AdamOptimizer를 이용해서 비용 함수를 최소화한다.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  
  # 정확도를 측정한다.
  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
      batch = mnist.train.next_batch(50)
      # 100 Step마다 training 데이터셋에 대한 정확도를 출력한다.
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
      # 50% 확률의 Dropout을 이용해서 학습을 진행한다.
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    # 테스트 데이터에 대한 정확도를 출력한다.
    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
