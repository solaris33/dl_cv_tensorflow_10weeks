# coding: utf-8

"""PTB 텍스트 파일을 파싱(parsing)하기 위한 유틸리티들(Utilities)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys

import tensorflow as tf


def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    if sys.version_info[0] >= 3:
      return f.read().replace("\n", "<eos>").split()
    else:
      return f.read().decode("utf-8").replace("\n", "<eos>").split()


def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None):
  """ "data_path"에 정의된 데이터 디렉토리로부터 raw PTB 데이터를 로드한다.
  PTB 텍스트 파일을 읽고, 문자열들(strings)을 정수 id값들(integer ids)로 변환한다.
  그리고 inputs을 mini-batch들로 나눈다.
  PTB 데이터셋은 아래의 Tomas Mikolov의 webpage에서 얻는다:
  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
  인자들(Args):
    data_path: simple-examples.tgz 파일의 압축이 해제된 디렉토리 경로 string
  반환값들(Returns):
    tuple (train_data, valid_data, test_data, vocabulary)
    각각의 data object 들은 PTBIterator로 전달(pass) 될 수 있다.
  """

  train_path = os.path.join(data_path, "ptb.train.txt")
  valid_path = os.path.join(data_path, "ptb.valid.txt")
  test_path = os.path.join(data_path, "ptb.test.txt")

  word_to_id = _build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, valid_data, test_data, vocabulary


def ptb_producer(raw_data, batch_size, num_steps, name=None):
  """raw PTB 데이터에 대해 반복한다.
  raw_data를 batches of examples로 변환하고 이 batches들로부터 얻은 Tensors를 반환한다.
  인자들(Args):
    raw_data: ptb_raw_data로부터 얻은 raw data outputs 중 하나.
    batch_size: int, 배치 크기(the batch size).
    num_steps: int, 학습하는 스텝의 크기(the number of unrolls).
    name: operation의 이름 (optional).
  반환값들(Returns):
    [batch_size, num_steps]로 표현된 Tensors 쌍(pair). tuple의 두번째 element는
    한 step만큼 time-shifted된 같은 데이터이다.
  에러값 발생(Raises):
    tf.errors.InvalidArgumentError: batch_size나 num_steps가 너무 크면 발생한다.
  """
  with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0 : batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.strided_slice(data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    x.set_shape([batch_size, num_steps])
    y = tf.strided_slice(data, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1])
    y.set_shape([batch_size, num_steps])
    return x, y
