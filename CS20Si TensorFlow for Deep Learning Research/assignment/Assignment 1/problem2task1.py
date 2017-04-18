"""
Starter code for logistic regression model to solve OCR task
with MNIST in TensorFlow
MNIST dataset: yann.lecun.com/exdb/mnist/
"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import time

# Define paramaters for the model
# learning_rate = 0.01
learning_rate = 0.005
batch_size = 128
n_epochs = 200

# Step 1: Read in data
# using TF Learn's built in function to load MNIST data to the folder data/mnist
mnist = input_data.read_data_sets('/data/mnist', one_hot=True)

# Step 2: create placeholders for features and labels
# each image in the MNIST data is of shape 28*28 = 784
# therefore, each image is represented with a 1x784 tensor
# there are 10 classes for each image, corresponding to digits 0 - 9.
X = tf.placeholder(shape=[batch_size, 784], dtype=tf.float32, name='X_placeholder')
Y = tf.placeholder(shape=[batch_size, 10], dtype=tf.float32, name='Y_placeholder')

# Step 3: create weights and bias
# weights and biases are initialized to 0
# shape of w depends on the dimension of X and Y so that Y = X * w + b
# shape of b depends on Y

# Network Parameters
w_1 = tf.Variable(tf.random_normal([784, 196], stddev=0.1), name='weights_for_layer1')
b_1 = tf.Variable(tf.zeros([1, 196]), name='bias_for_layer1')

w_2 = tf.Variable(tf.random_normal([196, 49]), name='weights_for_layer2')
b_2 = tf.Variable(tf.zeros([1, 49]), name='bias_for_layer2')

w_3 = tf.Variable(tf.random_normal([49, 10]), name='weights_for_layer3')
b_3 = tf.Variable(tf.zeros([1, 10]), name='bias_for_layer3')

# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer
# to get the probability distribution of possible label of the image
# DO NOT DO SOFTMAX HERE

# relu activation function
# n_epoch = 10 Accuracy = 0.9272
# n_epoch = 50 learning_rate = 0.005 Accuracy = 0.9397
# n_epoch = 50 learning_rate = 0.01 Accuracy = 0.9428
# n_epoch = 100 learning_rate = 0.01 Accuracy = 0.9503
# n_epoch = 200 learning_rate = 0.005 Accuracy = 0.9521

layer_1 = tf.nn.relu(tf.add(tf.matmul(X, w_1), b_1))
layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, w_2), b_2))
logits = tf.add(tf.matmul(layer_2, w_3), b_3)

# sigmoid activation function
# Accuracy 0.9223
# layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X, w_1), b_1))
# layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, w_2), b_2))
# logits = tf.add(tf.matmul(layer_2, w_3), b_3)

# relu activation function with dropout
# Accuracy 0.4097
# dropout = 0.5
# layer_1 = tf.nn.relu(tf.add(tf.matmul(X, w_1), b_1))
# layer_1 = tf.nn.dropout(layer_1, dropout)
# layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, w_2), b_2))
# layer_2 = tf.nn.dropout(layer_2, dropout)
# logits = tf.add(tf.matmul(layer_2, w_3), b_3)

# 4 layer sigmoid
# w_1 = tf.Variable(tf.random_normal([784, 392], stddev=0.1), name='weights_for_layer1')
# b_1 = tf.Variable(tf.zeros([1, 392]), name='bias_for_layer1')
#
# w_2 = tf.Variable(tf.random_normal([392, 49]), name='weights_for_layer2')
# b_2 = tf.Variable(tf.zeros([1, 49]), name='bias_for_layer2')
#
# w_3 = tf.Variable(tf.random_normal([49, 10]), name='weights_for_layer3')
# b_3 = tf.Variable(tf.zeros([1, 10]), name='bias_for_layer3')
#
# w_4 = tf.Variable(tf.random_normal([49, 10]), name='weights_for_layer4')
# b_4 = tf.Variable(tf.zeros([1, 10]), name='bias_for_layer4')
#
# layer_1 = tf.nn.relu(tf.add(tf.matmul(X, w_1), b_1))
# layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, w_2), b_2))
# layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, w_3), b_3))
# logits = tf.add(tf.matmul(layer_2, w_4), b_4)


# Step 5: define loss function
# use cross entropy loss of the real labels with the softmax of logits
# use the method:
# tf.nn.softmax_cross_entropy_with_logits(logits, Y)
# then use tf.reduce_mean to get the mean loss of the batch
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name='loss')
loss = tf.reduce_mean(entropy)

# Step 6: define training op
# using gradient descent to minimize loss

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


with tf.Session() as sess:
	start_time = time.time()
	sess.run(tf.global_variables_initializer())
	n_batches = int(mnist.train.num_examples/batch_size)
	for i in range(n_epochs): # train the model n_epochs times
		total_loss = 0

		for _ in range(n_batches):
			X_batch, Y_batch = mnist.train.next_batch(batch_size)
			# TO-DO: run optimizer + fetch loss_batch
			_, loss_batch = sess.run([optimizer, loss], feed_dict={X:X_batch, Y:Y_batch})
			total_loss += loss_batch
		print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))

	print('Final loss epoch {0}: {1}'.format(i, total_loss/n_batches))

	print('Optimization Finished!') # should be around 0.35 after 25 epochs

	# test the model
	n_batches = int(mnist.test.num_examples/batch_size)
	total_correct_preds = 0
	for i in range(n_batches):
		X_batch, Y_batch = mnist.test.next_batch(batch_size)
		_, loss_batch, logits_batch = sess.run([optimizer, loss, logits], feed_dict={X: X_batch, Y:Y_batch})
		preds = tf.nn.softmax(logits_batch)
		correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
		accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) # need numpy.count_nonzero(boolarr) :(
		total_correct_preds += sess.run(accuracy)

	print('Accuracy {0}'.format(total_correct_preds/mnist.test.num_examples))
