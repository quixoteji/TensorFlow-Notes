import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

def normalize_cols(m):
    col_max = m.max(axis = 0)
    col_min = m.min(axis = 0)
    return (m - col_min)/(col_max - col_min)


def Operational_Gates1():
    sess = tf.Session()
    a = tf.Variable(tf.constant(4.))
    x_val = 5.
    x_data = tf.placeholder(dtype=tf.float32)
    multiplication = tf.multiply(a, x_data)
    loss = tf.square(tf.subtract(multiplication, 50.))
    
    init = tf.initialize_all_variables()
    sess.run(init)
    opt = tf.train.GradientDescentOptimizer(0.01)
    train_step = opt.minimize(loss)
    print('Optimizing a Multiplication Gate Output to 50.')
    for i in range(10):
        # Run the train step
        sess.run(train_step, feed_dict={x_data: x_val})
        # Get the a and b values
        a_val = (sess.run(a))
        mult_output = sess.run(multiplication, feed_dict={x_data: x_val})
        print(str(a_val) + ' * ' + str(x_val) + ' = ' + str(mult_output))

def Operational_Gates2():
    sess = tf.Session()
    a = tf.Variable(tf.constant(1.0))
    b = tf.Variable(tf.constant(1.0))
    x_val = 5.
    x_data = tf.placeholder('float32')

    two_gate = tf.add(tf.multiply(a, x_data), b)
    loss = tf.square(tf.subtract(two_gate, 50.))

    opt = tf.train.GradientDescentOptimizer(0.01)
    train_step = opt.minimize(loss)

    init = tf.initialize_all_variables()
    sess.run(init)

    print('Optimizing Two Gate Output to 50.')
    for i in range(10):
        # Run the train step
        sess.run(train_step, feed_dict={x_data: x_val})
        a_val, b_val = (sess.run(a), sess.run(b))
        two_gate_output = sess.run(two_gate, feed_dict={x_data: x_val})
        print(str(a_val) + '*' + str(x_val) + '+' + str(b_val) + ' = ' + str(two_gate_output))

def Activations():
    sess = tf.Session()
    tf.set_random_seed(5)
    np.random.seed(42)
    batch_size = 50
    a1 = tf.Variable(tf.random_normal(shape=[1,1]))
    b1 = tf.Variable(tf.random_uniform(shape=[1,1]))
    a2 = tf.Variable(tf.random_normal(shape=[1,1]))
    b2 = tf.Variable(tf.random_uniform(shape=[1,1]))
    x = np.random.normal(2, 0.1, 500)
    x_data = tf.placeholder(shape=[None, 1], dtype = 'float32')

    sigmoid_activation = tf.sigmoid(tf.add(tf.matmul(x_data, a1), b1))
    relu_activation = tf.nn.relu(tf.add(tf.matmul(x_data, a2), b2))

    loss1 = tf.reduce_mean(tf.square(tf.subtract(sigmoid_activation, 0.75)))
    loss2 = tf.reduce_mean(tf.square(tf.subtract(relu_activation, 0.75)))

    opt = tf.train.GradientDescentOptimizer(0.01)
    train_step_sigmoid = opt.minimize(loss1)
    train_step_relu = opt.minimize(loss2)

    init = tf.global_variables_initializer()
    sess.run(init)

    loss_vec_sigmoid = []
    loss_vec_relu = []
    activation_sigmoid = []
    activation_relu = []
    for i in range(750):
        rand_indices = np.random.choice(len(x), size = batch_size)
        x_vals = np.transpose([x[rand_indices]])
        sess.run(train_step_sigmoid, feed_dict = {x_data: x_vals})
        sess.run(train_step_relu, feed_dict = {x_data: x_vals})
        loss_vec_sigmoid.append(sess.run(loss1, feed_dict={x_data: x_vals}))
        loss_vec_relu.append(sess.run(loss2, feed_dict={x_data: x_vals}))
        activation_sigmoid.append(np.mean(sess.run(sigmoid_activation, feed_dict={x_data: x_vals})))
        activation_relu.append(np.mean(sess.run(relu_activation, feed_dict={x_data: x_vals})))

    plt.plot(activation_sigmoid, 'k-', label = 'Sigmoid Activation')
    plt.plot(activation_relu, 'r--', label = 'Relu Activation')
    plt.ylim([0, 1.0])
    plt.title('Activation Outputs')
    plt.xlabel('Generation')
    plt.ylabel('Outputs')
    plt.legend(loc = 'upper right')
    plt.show()
    plt.ylim([0, 1.0])
    plt.plot(loss_vec_sigmoid, 'k-', label = 'Sigmoid Loss')
    plt.plot(loss_vec_relu, 'r--', label = 'Relu Loss')
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

def One_Layer_NN():
    iris = datasets.load_iris()
    x_vals = np.asarray([x[0:3] for x in iris.data])
    y_vals = np.asarray([x[3] for x in iris.data])

    sess = tf.Session()
    seed = 2 
    tf.set_random_seed(seed)
    np.random.seed(seed)

    train_indicies = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace = False)
    test_indicies = np.asarray(list(set(range(len(x_vals)))-set(train_indicies)))
    x_vals_train = x_vals[train_indicies]
    x_vals_test = x_vals[test_indicies]
    y_vals_train = y_vals[train_indicies]
    y_vals_test = np.transpose([y_vals[test_indicies]])

    x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
    x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

    batch_size = 50
    x_data = tf.placeholder(shape=[None, 3], dtype='float32')
    y_data = tf.placeholder(shape=[None, 1], dtype='float32')

    # Hidden_layer
    # hidden_layer_nodes = 5
    hidden_layer_nodes = 15
    A1 = tf.Variable(tf.random_normal(shape = [3, hidden_layer_nodes]))
    b1 = tf.Variable(tf.random_normal(shape = [hidden_layer_nodes]))
    A2 = tf.Variable(tf.random_normal(shape = [hidden_layer_nodes, 1]))
    b2 = tf.Variable(tf.random_normal(shape = [1]))

    hidden_output = tf.nn.relu(tf.add(tf.matmul(x_data, A1), b1))
    final_output = tf.nn.relu(tf.add(tf.matmul(hidden_output, A2), b2))

    loss = tf.reduce_mean(tf.square(y_data - final_output))

    opt = tf.train.GradientDescentOptimizer(0.005)
    train_step = opt.minimize(loss)
    init = tf.global_variables_initializer()
    sess.run(init)

    loss_vec = []
    test_loss = []
    for i in range(500):
        rand_index = np.random.choice(len(x_vals_train), size = batch_size)
        rand_x = x_vals_train[rand_index]
        rand_y = np.transpose([y_vals_train[rand_index]])

        sess.run(train_step, feed_dict={x_data: rand_x, y_data: rand_y})
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_data: rand_y})
        loss_vec.append(np.sqrt(temp_loss))

        temp_test_loss = sess.run(loss, feed_dict = {x_data: x_vals_test, y_data: y_vals_test})
        test_loss.append(np.sqrt(temp_test_loss))

        if (i+1)%50==0:
            print('Generation: ' + str(i+1) + '. Loss = ' + str(temp_loss))

    plt.plot(loss_vec, 'k-', label = 'Training Loss')
    plt.plot(test_loss, 'r--', label = 'Testing Loss')
    plt.title('MSE Loss per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()


# def Different_Layer():
# TensorFlow input data 4D = [batch_size, width, height, channels]
def conv_layer_1d(input_1d, my_filter):
    # Make 1d input into 4d
    input_2d = tf.expand_dims(input_1d, 0)
    input_3d = tf.expand_dims(input_2d, 0)
    input_4d = tf.expand_dims(input_3d, 3)
    # Perform convolution
    convolution_output = tf.nn.conv2d(input_4d, filter = my_filter, strides=[1,1,1,1], padding='VALID')
    # Now drop extra dimensions
    conv_output_1d = tf.squeeze(convolution_output)
    return conv_layer_1d

def activation(input_1d):
    return tf.nn.relu()

def OneD_Convolution():
    sess = tf.Session()
    data_size = 25
    data_1d = np.random.normal(size = data_size)
    x_input_1d = tf.placeholder(dtype=tf.float32, shape=[data_size])
    my_filter = tf.Variable(tf.random_normal(shape=[1,5,1,1]))
    my_convolution_output = conv_layer_1d(x_input_1d, my_filter)







    

if __name__ == '__main__':
    while(True):
        print('Choose one method to conduct : ')
        print('1.Operational Gates - one input')
        print('2.Operational Gates - two inputs')
        print('3. Activations')
        print('4. One Layer Neural Network')
        # print('5. Demming Regression')
        # print('6. Lasso and Ridge Regression')
        # print('7. Elastic Net Regression')
        # print('8. Logistic Regression')
        x = int(input())
        if (x == 1):
            Operational_Gates1()
        elif (x == 2):
            Operational_Gates2()
        elif (x == 3):
            Activations()
        elif (x == 4):
            One_Layer_NN()
        # elif (x == 5):
        #     DemingRegression()
        # elif (x == 6):
        #     Lasso_Ridge_Regression()
        # elif (x == 7):
        #     Elastic_Net_Regression()
        # elif (x == 8):
        #     Logistic_Regression()
        else:
            print('ERROR! Choose another method!')

        

