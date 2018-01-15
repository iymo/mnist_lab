import os
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import sys
import urllib.request
import pygame
import time
import wave
from scipy import ndimage


tf.set_random_seed(777)
from tensorflow.examples.tutorials.mnist import input_data



######인공지능 MNIST

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
nb_classes = 10
## MNIST 변수와 가설식
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])
W = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

#피라메터
training_epochs = 15
batch_size = 100

## 학습과정
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={
                            X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1),
              'cost =', '{:.9f}'.format(avg_cost))

    print("Learning finished")

    # Test the model using test sets
    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={
          X: mnist.test.images, Y: mnist.test.labels}))

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("Prediction: ", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

    # 에러가 발생하면 보여준다..
    plt.imshow(mnist.test.images[r:r + 1].reshape(28, 28), cmap='Greys',interpolation='nearest')
    plt.show()

    ####### 음성인식
    client_id = "kLjan7Utur6FdY7obBnK"
    client_secret = "yvapa0q6yv"
    t=sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]})
    s= str(t)
    k='숫자는'+s+'인것 같습니다. 숫자는'+s+'입니다.'
    encText = urllib.parse.quote(k)
    data = "speaker=jinho&speed=0&text=" + encText;
    url = "https://openapi.naver.com/v1/voice/tts.bin"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id",client_id)
    request.add_header("X-Naver-Client-Secret",client_secret)
    response = urllib.request.urlopen(request, data=data.encode('utf-8'))
    rescode = response.getcode()
    if(rescode==200):
        print("TTS wav 저장")
        response_body = response.read()
        with open('test.wav', 'wb') as f:
            f.write(response_body)

    else:
        print("Error Code:" + rescode)


    #def talk():
     #   global sounds
      #  sounds = pygame.mixer.Sound('test.wav')




