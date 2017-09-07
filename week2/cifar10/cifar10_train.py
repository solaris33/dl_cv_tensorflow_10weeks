# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf

import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
#tf.app.flags.DEFINE_integer('max_steps', 1000000,
#                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")


def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    ## 이미지와 레이블을 가져오는데, 디스토트할지 말지 결정할 수 있다.
    ## 0번째 cpu에서 하는데
    ## 그 이유는 GPU랑 같이 쓸때는 
    ## 데이터 읽어들이는 덜 중요한 것은 cpu이용하자는 것.
    with tf.device('/cpu:0'):
      images, labels = cifar10.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images)

    # Calculate loss.
    loss = cifar10.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.train(loss, global_step)

    ## 로저훅. 텐서플로우 많이 해봐야 아는데
    ## 이걸 쓰면 한스텝 지날때마다 그 값을 지정할 수 있는 코드임.
    ## begin, before_run, after_run
    ## begin: 처음 에폭 시작할때 진입하는 부분. -1에서 시작. 스타트타임 잡고
    ## before_run: 한스텝 업뎃하고.. 
    ## after_run: 커런트 타임 저장하고 한 에폭 도는데 얼마나 걸렸나 측정.
    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

    ## 확장된 형태의 세션
    ## 로저 훅 지정한 것을..
    ## 훅에 인풋으로 넣어줘서.. 
    ## NaNTensorHook은 컴터 연산과정에서 불가능한 값이 나오는  0으로 나누는.
    ## loss 할때 그런게 나오면 바로 스탑하겠다라는 것.
    ## 이걸 사용하는 이유는 그냥 세션 썼는데 
    ## 모니터링 세션 쓰면 중간중간에 로그 찍을 수 잇고.
    ## 발전된 형태임 우왕 글쿠나.
    ## 맥스스텝에 도달하면 정지하겠다.
    ## ConfigProto가 트루라면 모든 연산이 어떤 cpu, gpu에서 도는지 확인할 수 있다.
    ## 혹시 그럼 프로덕션 환경에서는 끄는 걸까? 굳이 안그래도 된다고 함. NaN처리 같은거 더 효율적. 로저 훅도 그렇고.
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run(train_op)


def main(argv=None):  # pylint: disable=unused-argument
  ## 없으면 다운로드
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  ## 학습시작.
  train()


if __name__ == '__main__':
  tf.app.run()
