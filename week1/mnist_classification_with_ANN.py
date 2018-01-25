# -*- coding: utf-8 -*-

# 텐서플로우를 이용한 ANN(Artificial Neural Networks) 구현

# python3와 호환성을 위한 임포트
from __future__ import print_function

# 텐서플로우 라이브러리를 임포트한다.
import tensorflow as tf

# MNIST 데이터를 다운로드 한다.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# 학습을 위한 하이퍼 파라미터들
learning_rate = 0.001
training_epochs = 30    # 학습횟수
batch_size = 256        # 배치개수
display_step = 1        # 손실함수 출력 횟수 

# 네트워크 구조 지정하기 위한 하이퍼파라미터들
input_size = 784
hidden1_size = 256
hidden2_size = 256
output_size = 10

# 입력값과 출력값을 받기 위한 플레이스홀더
x = tf.placeholder(tf.float32, [None, input_size])
true_y = tf.placeholder(tf.float32, [None, output_size])

# ANN 모델 구현
def build_ANN(x):
    W1 = tf.Variable(tf.random_normal([input_size, hidden1_size]))
    b1 = tf.Variable(tf.random_normal([hidden1_size]))
    L1_output = tf.nn.relu(tf.matmul(x,W1) + b1)
    W2 = tf.Variable(tf.random_normal([hidden1_size, hidden2_size]))
    b2 = tf.Variable(tf.random_normal([hidden2_size]))
    L2_output = tf.nn.relu(tf.matmul(L1_output,W2) + b2)
    W_output = tf.Variable(tf.random_normal([hidden2_size, output_size]))
    b_output = tf.Variable(tf.random_normal([output_size]))
    output = tf.matmul(L2_output,W_output) + b_output

    return output

# ANN 모델 구현
predicted_value = build_ANN(x)

# 손실함수와 옵티마이저를 정의한다.
# tf.nn.softmax_cross_entropy_with_logits 함수를 이용하여 마지막에 softmax 함수를 적용한다.
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predicted_value, labels=true_y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# 세션을 열고 그래프를 실행한다.
with tf.Session() as sess:
    # 변수들에 초기값을 할당한다.
    sess.run(tf.global_variables_initializer())

    # 지정된 횟수만큼 학습을 진행한다.
    for epoch in range(training_epochs):
        average_loss = 0.
        # 전체 배치를 불러온다.
        total_batch = int(mnist.train.num_examples/batch_size)
        # 모든 배치들에 대해서 최적화를 수행한다.
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # 옵티마이저를 실행해서 파라마터들을 업데이트한다.
            _, current_loss = sess.run([optimizer, loss], feed_dict={x: batch_x, true_y: batch_y})
            # 평균 손실을 측정한다.
            average_loss += current_loss / total_batch
        # 지정된 epoch마다 학습결과를 출력한다.
        if epoch % display_step == 0:
            print("Epoch:", (epoch+1), ", 손실 함수(loss)=", \
                "{:.9f}".format(average_loss))
    print("최적화 끝!")

    # 테스트 데이터를 이용해서 학습된 모델이 얼마나 정확한지 정확도를 출력한다.
    correct_prediction = tf.equal(tf.argmax(predicted_value, 1), tf.argmax(true_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("정확도(Accuracy):", accuracy.eval({x: mnist.test.images, true_y: mnist.test.labels}))
