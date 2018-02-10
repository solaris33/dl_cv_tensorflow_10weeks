# -*- coding: utf-8 -*-
# AutoEncoder를 이용한 MNIST Reconstruction

from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# MNIST 데이터를 다운로드한다.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 파라미터들
learning_rate = 0.01
training_epochs = 20    # 반복횟수
batch_size = 256        # 배치개수
display_step = 1
examples_to_show = 10

# 네트워크 구조를 정의한다.
n_hidden_1 = 300 
n_hidden_2 = 150
n_input = 784 

# 입력을 받기 위한 플레이스홀더를 정의한다.
X = tf.placeholder(tf.float32, [None, n_input])

def build_autoencoder(x):
    # 인코딩(Encoding) - 784 -> 300 -> 150
    W1 = tf.Variable(tf.random_normal([n_input, n_hidden_1]))
    b1 = tf.Variable(tf.random_normal([n_hidden_1]))
    L1 = tf.nn.sigmoid(tf.matmul(x,W1) + b1)
    W2 = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]))
    b2 = tf.Variable(tf.random_normal([n_hidden_2]))
    L2 = tf.nn.sigmoid(tf.matmul(L1,W2) + b2)
    # 디코딩(Decoding) 150 -> 300 -> 784
    W3 = tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1]))
    b3 = tf.Variable(tf.random_normal([n_hidden_1]))
    L3 = tf.nn.sigmoid(tf.matmul(L2,W3) + b3)
    W4 = tf.Variable(tf.random_normal([n_hidden_1, n_input]))
    b4 = tf.Variable(tf.random_normal([n_input]))
    reconstructed_x = tf.nn.sigmoid(tf.matmul(L3,W4) + b4)

    return reconstructed_x

# 오토인코더를 그래프 구조를 정의한다.
y_pred = build_autoencoder(X)
# 타겟데이터는 인풋데이터와 같다. 
y_true = X

# 손실함수와 옵티마이저를 정의한다.
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# 그래프를 실행한다.
with tf.Session() as sess:
    # 변수들의 초기값을 할당한다.
    sess.run(tf.global_variables_initializer())
    total_batch = int(mnist.train.num_examples/batch_size)
    # 트레이닝 횟수만큼 학습을 진행한다.
    for epoch in range(training_epochs):
        # 전체 배치에 대해서 학습을 진행한다.
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, loss], feed_dict={X: batch_xs})
        # 지정된 주기마다 학습 결를를 출력한다.
        if epoch % display_step == 0:
            print("반복횟수(Epoch) :", '%04d' % (epoch+1),
                  "손실함수(loss) =", "{:.9f}".format(c))

    print("최적화 끝!")

    # 테스트 데이터로 Reconstruction을 수행한다.
    reconstructed_result = sess.run(y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
    # 원본 MNIST 데이터와 Reconstruction 결과를 비교한다.
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(reconstructed_result[i], (28, 28)))
    f.show()
    plt.draw()
    plt.waitforbuttonpress()
