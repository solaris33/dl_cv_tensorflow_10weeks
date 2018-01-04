# -*- coding: utf-8 -*-

"""텐서보드에서 요약정보(summaries)를 보여주는 간단한 MNIST 분류기

이 소스는 별로 특별하지 않은 MNIST 모델이다. 하지만, 텐서보드 그래프 탐색기에서 그래프를 읽을수 있게
tf.name_scope를 사용하고, 텐서보드에서 의미있게 그룹핑되어 보여질 수 있도록 요약 태그(summary tag)
를 사용하는 방법을 배우기 위한 좋은 예제이다.

이 예제는 텐서보드 대시보드의 기능들을 보여준다.
"""

# 절대 임포트 설정
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 필요한 라이브러리들을 임포트
import argparse
import sys

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None


def train():
  # 데이터 입력
  mnist = input_data.read_data_sets(FLAGS.data_dir,
                                    one_hot=True,
                                    fake_data=FLAGS.fake_data)

  sess = tf.InteractiveSession()
  # 멀티레이어 모델 생성

  # 입력 placeholders
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

  with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)

  # 가중치 변수들은 0으로 초기화할 수 없다. -그러면 네트워크 학습이 제대로 진행되지 않는다.
  def weight_variable(shape):
    """적절히 초기화한 가중치 변수들을 생성한다."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(shape):
    """적절히 초기화한 바이어스(bias) 변수들을 생성한다."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def variable_summaries(var):
    """텐서에 많은 양의 요약정보(summaries)를 붙인다. (텐서보드 시각화를 위해서)"""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

  def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """간단한 neural net layer를 만들기 위해 재사용(resuable)이 가능한 코드

    행렬 곱셈(matrix multiply), 바이어스 덧셈(bias add) 이후에 nonlinearize를 위해 ReLU를 사용한다.
    또한, 그래프를 읽기 쉽게 만들기 위해 name scoping을 지정하고 summary ops들을 추가한다.
    """
    # 그래프를 논리적으로 그룹핑(grouping)하기 위해 name scope를 추가한다.
    with tf.name_scope(layer_name):
      # 이 변수들은 레이어의 가중치들의 상태를 가지고(hold) 있을 것이다.
      with tf.name_scope('weights'):
        weights = weight_variable([input_dim, output_dim])
        variable_summaries(weights)
      with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
        variable_summaries(biases)
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.summary.histogram('pre_activations', preactivate)
      activations = act(preactivate, name='activation')
      tf.summary.histogram('activations', activations)
      return activations

  hidden1 = nn_layer(x, 784, 500, 'layer1')

  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)

  # softmax 활성함수는 아직 적용하지 않는다. (아래를 참조하라.)
  y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

  with tf.name_scope('cross_entropy'):
    # cross_entropy의 수학적 표현(formulation)
    #
    # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
    #                               reduction_indices=[1]))
    #
    # 위 함수는 계산적으로(numerically) 불안정(unstable)할 수 있다. 
    #
    # 따라서 여기서 우리는, nn_layer의 raw output에 대해 tf.nn.softmax_cross_entropy_with_logits을 사용한다.
    # 그리고 나서 배치(batch)간의 평균을을 취한다.
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    with tf.name_scope('total'):
      cross_entropy = tf.reduce_mean(diff)
  tf.summary.scalar('cross_entropy', cross_entropy)

  with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
        cross_entropy)

  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)

  # 모든 요약정보들(summaries)을 합치고(merge) 그들을 지정된 경로에 쓴다.(write) (기본경로: /tmp/tensorflow/mnist/logs/mnist_with_summaries)
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
  tf.global_variables_initializer().run()

  # 모델을 학습시키고, 요약정보들(summaries)들을 쓴다.
  # 매 10 step마다, test_set accuracy를 측정하고, test suammries을 쓴다.
  # 그외의 step에서는, 트레이닝 데이터에 대해서 train-step을 실행하고, training summaries을 추가한다.

  def feed_dict(train):
    """텐서플로우 feed_dict를 만든다: 데이터를 Tensor placeholders에 맵핑(map)한다."""
    if train or FLAGS.fake_data:
      xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
      k = FLAGS.dropout
    else:
      xs, ys = mnist.test.images, mnist.test.labels
      k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

  for i in range(FLAGS.max_steps):
    if i % 10 == 0:  # summaries와 test-set accuracy를 기록(record)한다.
      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
      test_writer.add_summary(summary, i)
      print('Accuracy at step %s: %s' % (i, acc))
    else:  # train set summaries를 기록하고, 학습을 진행한다.
      if i % 100 == 99:  # 실행 상태들(execution stats)을 기록한다.
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict(True),
                              options=run_options,
                              run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)
        print('Adding run metadata for', i)
      else:  # summary를 기록한다.
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)
  train_writer.close()
  test_writer.close()


def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
  parser.add_argument('--max_steps', type=int, default=1000,
                      help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  parser.add_argument('--log_dir', type=str, default='/tmp/tensorflow/mnist/logs/mnist_with_summaries',
                      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
