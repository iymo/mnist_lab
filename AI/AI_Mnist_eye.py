import os
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import urllib.request
import pygame
import time
import wave
from scipy import ndimage
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import math
import cv2
import random
from PIL import Image
tf.set_random_seed(777)


# 이미지 조정
def getBestShift(img):
    cy, cx = ndimage.measurements.center_of_mass(img)

    rows, cols = img.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)

    return shiftx, shifty


# 이미지 조정
def shift(img, sx, sy):
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, M, (cols, rows))
    return shifted


######인공지능 MNIST

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
nb_classes = 10
## MNIST 변수와 가설식
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])  # 실제값 우리가 훈련하기를 원하는 숫자
W = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# 피라메터
training_epochs = 25
batch_size = 90

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
#### 사진 캡처
    cam = cv2.VideoCapture(0)
    cam.set(3, 28)
    cam.set(4, 28)
    cv2.namedWindow("test")



    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("test", frame)
        cv2.imshow('test', frame)
        if not ret:
            break
        k = cv2.waitKey(1)

        if k % 256 == 27:
                # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            img_name = "blog/number.png"
            cv2.imwrite(img_name, gray)
            print("사진출력 완료 esc를 누르세요")


    cam.release()

    cv2.destroyAllWindows()
    #픽셀처리

    # Get one and predict
    # 사진을 저장 할 수 있는 배열
    images = np.zeros((1, 784))
    # 정확한 값.
    correct_vals = np.zeros((1, 10))

    gray = cv2.imread("blog/number.png", 0)
    gray = cv2.resize(255 - gray, (28, 28))
    (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    while np.sum(gray[0]) == 0:
        gray = gray[1:]

    while np.sum(gray[:, 0]) == 0:
        gray = np.delete(gray, 0, 1)

    while np.sum(gray[-1]) == 0:
        gray = gray[:-1]

    while np.sum(gray[:, -1]) == 0:
        gray = np.delete(gray, -1, 1)

    rows, cols = gray.shape

    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
        # first cols than rows
        gray = cv2.resize(gray, (cols, rows))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))
        # first cols than rows
        gray = cv2.resize(gray, (cols, rows))

    colsPadding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
    rowsPadding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
    gray = np.lib.pad(gray, (rowsPadding, colsPadding), 'constant')

    shiftx, shifty = getBestShift(gray)
    shifted = shift(gray, shiftx, shifty)
    gray = shifted

    # 처리된 이미지 저장
    i=0
    cv2.imwrite("pro-img/number.png", gray)
    flatten = gray.flatten() / 255.0
    images[i] = flatten
    correct_val = np.zeros((10))
    correct_vals[i] = correct_val
    i += 1

    prediction = tf.argmax(hypothesis, 1)
    print("prediction",sess.run(prediction, feed_dict={X: images}))


###음성인식
    client_id = "kLjan7Utur6FdY7obBnK"
    client_secret = "yvapa0q6yv"
    t = sess.run(prediction, feed_dict={X: images})
    s = str(t)
    k = '숫자는' + s + '인것 같습니다. 숫자는' + s + '입니다.'
    encText = urllib.parse.quote(k)
    data = "speaker=jinho&speed=0&text=" + encText;
    url = "https://openapi.naver.com/v1/voice/tts.bin"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", client_id)
    request.add_header("X-Naver-Client-Secret", client_secret)
    response = urllib.request.urlopen(request, data=data.encode('utf-8'))
    rescode = response.getcode()
    if (rescode == 200):
        print("TTS wav 저장")
        response_body = response.read()
        with open('test.wav', 'wb') as f:
            f.write(response_body)

    else:
         print("Error Code:" + rescode)

    os.system("test.wav")
    # def talk():
    #   global sounds
    #  sounds = pygame.mixer.Sound('test.wav')




