"""
Codes for Problem 2 Task 3
Loading csv files to tensorflow
"""

import tensorflow as tf
import time

# Parameters
DATA_PATH_TESTING = 'testing.csv'
DATA_PATH_TRAINING = 'training.csv'
BATCH_SIZE = 40
N_FEATURES = 9
LEARNING_RATE = 0.1
N_EPOCHS = 20

def batch_generator(filenames, batch_size=BATCH_SIZE):
    """
    filenames is the list of files you want to read from.
    """
    filename_queue = tf.train.string_input_producer(filenames)
    reader = tf.TextLineReader(skip_header_lines=1)
    # Returns the next record (key, value pair) produced by a reader
    _, value = reader.read(filename_queue)

    # Set default for data queue
    record_defaults = [[1.0] for _ in range(N_FEATURES)]
    record_defaults[4] = ['']
    record_defaults.append([1])

    # Read in 10 rows of data
    content = tf.decode_csv(value, record_defaults=record_defaults)

    # Convert 5th column(present/absent) to binary value 0 and 1
    condition = tf.equal(content[4], tf.constant('Present'))
    content[4] = tf.where(condition, tf.constant(1.0), tf.constant(0.0))

    # pack features
    features = tf.stack(content[:N_FEATURES])

    # assign label
    label = content[-1]

    # minimum number elements in the queue after a dequeue, used to ensure
    # that the samples are sufficiently mixed
    # I think 10 times the BATCH_SIZE is sufficient
    min_after_dequeue = 10 * batch_size

    # the maximum number of elements in the queue
    capacity = 20 * batch_size

    # shuffle the data to generate BATCH_SIZE sample pairs
    data_batch, label_batch = tf.train.shuffle_batch([features, label],\
                                                batch_size=batch_size,\
                                                capacity=capacity,\
                                                min_after_dequeue=min_after_dequeue)
    return data_batch, label_batch

def generate_batches(data_batch, label_batch):
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        features, labels = sess.run([data_batch, label_batch])
        coord.request_stop()
        coord.join(threads)
        return features, labels

def main():

    X = tf.placeholder(shape=[BATCH_SIZE, 9], dtype=tf.float32, name='X_placeholder')
    Y = tf.placeholder(shape=[BATCH_SIZE,1], dtype=tf.float32, name='Y_placeholder')

    W = tf.Variable(tf.random_normal([9, 1], stddev=0.1), name='weights')
    b = tf.Variable(tf.zeros(1), name='bias')

    # Building Model
    Y_pred = tf.add(tf.matmul(X, W), b)

    # loss
    # loss = tf.square(Y - Y_pred)
    # logistic regression, not linear regression
    # Y_pred = tf.matrix_transpose(Y_pred)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_pred, labels=Y))
    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

    # Train model
    with tf.Session() as sess:
        start_time = time.time()
        sess.run(tf.global_variables_initializer())
        n_batches = int(400/BATCH_SIZE)
        for i in range(N_EPOCHS):
            total_loss = 0
            training_data_batch, training_label_batch = \
                batch_generator([DATA_PATH_TRAINING], batch_size=BATCH_SIZE)
            testing_data, testing_label = \
                batch_generator([DATA_PATH_TESTING], batch_size=1)
            for _ in range(n_batches):
                # print(_)
                X_batch, Y_batch = generate_batches(training_data_batch, \
                                                    training_label_batch)
                # print(X_batch, Y_batch)
                # print(Y1=sess.run(Y_pred))
                _, loss_batch = sess.run([optimizer, loss], feed_dict={X: X_batch, Y:Y_batch})
                # print(Y_1, Y_batch)
                total_loss += loss_batch
                # print(total_loss)
            print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))
        print('Total time: {0} seconds'.format(time.time() - start_time))
        print('Optimization Finished!')
        # Test models
        # n_batches = int(mnist.test.num_examples/batch_size)
        # total_correct_preds = 0
        # for i in range(n_batches):
        #     X_batch, Y_batch = mnist.test.next_batch(batch_size)
        #     _, loss_batch, logits_batch = sess.run([optimizer, loss, logits], feed_dict={X: X_batch, Y:Y_batch})
        #     preds = tf.nn.softmax(logits_batch)
        #     correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
        #     accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) # need numpy.count_nonzero(boolarr) :(
        #     total_correct_preds += sess.run(accuracy)
        # print('Accuracy {0}'.format(total_correct_preds/mnist.test.num_examples))

if __name__ == '__main__':
    main()
