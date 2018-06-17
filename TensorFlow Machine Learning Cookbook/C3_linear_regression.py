import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
import requests
from sklearn.preprocessing import normalize

def load_data():
    '''
    x = (A^TA)^(-1)A^Tb
    '''
    x_vals = np.linspace(0, 10, 100)
    y_vals = x_vals + np.random.normal(0, 1, 100)

    x_vals_column = np.transpose(np.matrix(x_vals))
    ones_column = np.transpose(np.matrix(np.repeat(1,100)))

    A = np.column_stack((x_vals_column, ones_column))
    b = np.transpose(np.matrix(y_vals))

    A_tensor = tf.constant(A)
    b_tensor = tf.constant(b)
    return A_tensor, b_tensor, x_vals, y_vals

def plot(x_vals, y_vals, slop, y_intercept):
    best_fit = []
    for i in x_vals:
        best_fit.append(slop*i+y_intercept)
    plt.plot(x_vals, y_vals, 'o', label = 'Data')
    plt.plot(x_vals, best_fit, 'r-', label = 'Best fit line', linewidth = 4)
    plt.legend(loc = 'upper left')
    plt.show()

def Matrix_Inverse_Method():
    sess = tf.Session()
    A_tensor, b_tensor, x_vals, y_vals = load_data()
    tA_A = tf.matmul(tf.transpose(A_tensor), A_tensor)
    tA_A_inv = tf.matrix_inverse(tA_A)
    product = tf.matmul(tA_A_inv, tf.transpose(A_tensor))
    solution = tf.matmul(product, b_tensor)
    solution_eval = sess.run(solution)

    slope = solution_eval[0][0]
    y_intercept = solution_eval[1][0]
    print('slope: ' + str(slope))
    print('y_intercept: ' + str(y_intercept))
    plot(x_vals, y_vals, slope, y_intercept)

def Decomposing_Method():
    '''
    Ax = b
    LL'x = b
    1st: Ly = b
    2nd: L'x = y
    '''
    A_tensor, b_tensor, x_vals, y_vals = load_data()
    sess = tf.Session()
    tA_A = tf.matmul(tf.transpose(A_tensor), A_tensor)
    L = tf.cholesky(tA_A)
    tA_b = tf.matmul(tf.transpose(A_tensor), b_tensor)
    sol1 = tf.matrix_solve(L, tA_b)
    sol2 = tf.matrix_solve(tf.transpose(L), sol1)

    solution_eval = sess.run(sol2)
    slope = solution_eval[0][0]
    y_intercept = solution_eval[1][0]
    print('slope: ' + str(slope))
    print('y_intercept: ' + str(y_intercept))
    plot(x_vals, y_vals, slope, y_intercept)

def TensorFlow_Method():
    tf.reset_default_graph()
    sess = tf.Session()
    iris = datasets.load_iris()
    x_vals = np.array([x[3] for x in iris.data])
    y_vals = np.array([y[0] for y in iris.data])

    learning_rate = 0.05
    batch_size = 25

    x_data = tf.placeholder(shape = [None, 1], dtype = tf.float32)
    y_target = tf.placeholder(shape = [None, 1], dtype = tf.float32)
    A = tf.Variable(tf.random_normal(shape = [1,1]))
    b = tf.Variable(tf.random_normal(shape = [1,1]))

    model_output = tf.add(tf.matmul(x_data, A), b)

    loss = tf.reduce_mean(tf.square(y_target - model_output))
    init = tf.global_variables_initializer()
    sess.run(init)
    my_opt = tf.train.GradientDescentOptimizer(learning_rate)
    train_step = my_opt.minimize(loss)

    loss_vec = []
    for i in range(100):
        rand_index = np.random.choice(len(x_vals), size = batch_size)
        rand_x = np.transpose([x_vals[rand_index]])
        rand_y = np.transpose([y_vals[rand_index]])
        sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        loss_vec.append(temp_loss)

        if (i+1)%25 == 0:
            print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)) + 'b = ' + str(sess.run(b)))
            print('Loss = ' + str(temp_loss))

    [slope] = sess.run(A)
    [y_intercept] = sess.run(b)

    best_fit = []
    for i in x_vals:
        best_fit.append(slope * i + y_intercept)

    plt.plot(x_vals, y_vals, 'o', label = 'Data Points')
    plt.plot(x_vals, best_fit, 'r-', label = 'Best fit line', linewidth = 4)
    plt.legend(loc = 'upper left')
    plt.title('Sepal Length vs Pedal Width')
    plt.xlabel('Pedal Width')
    plt.ylabel('Sepal Length')
    plt.show()

    plt.plot(loss_vec, 'k-')
    plt.title('L2 Loss per Generation')
    plt.xlabel('Generation')
    plt.ylabel('L2 Loss')
    plt.show()

    
def Loss_Comparation():
    # sess = tf.Session()
    # sess1 = tf.Session()
    # sess2 = tf.Session()
    iris = datasets.load_iris()
    x_vals = np.array([x[3] for x in iris.data])
    y_vals = np.array([y[0] for y in iris.data])

    batch_size = 5
    learning_rate = 0.1
    iterations = 100

    x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    A = tf.Variable(tf.random_normal(shape=[1,1]))
    b = tf.Variable(tf.random_normal(shape=[1,1]))
    model_output = tf.add(tf.matmul(x_data, A), b)

    loss_l1 = tf.reduce_mean(tf.abs(y_target - model_output))
    loss_l2 = tf.reduce_mean(tf.square(y_target - model_output))
    ops_l1 = tf.train.GradientDescentOptimizer(0.4)
    ops_l2 = tf.train.GradientDescentOptimizer(0.4)
    train_step_l1 = ops_l1.minimize(loss_l1)
    train_step_l2 = ops_l2.minimize(loss_l2)

    
    los_vec_l1 = []
    los_vec_l2 = []
    sess1 = tf.Session()
    init1 = tf.global_variables_initializer()
    sess1.run(init1)
    for i in range(iterations):
        rand_index = np.random.choice(len(iris), size = batch_size)
        rand_x = np.transpose([x_vals[rand_index]])
        rand_y = np.transpose([y_vals[rand_index]])
        sess1.run(train_step_l1, feed_dict={x_data: rand_x, y_target: rand_y})
        temp_loss_l1 = sess1.run(loss_l1, feed_dict={x_data: rand_x, y_target: rand_y})
        los_vec_l1.append(temp_loss_l1)
    sess1.close()

    sess2 = tf.Session()
    init2 = tf.global_variables_initializer()
    sess2.run(init2)
    for i in range(iterations):
        rand_index = np.random.choice(len(iris), size = batch_size)
        rand_x = np.transpose([x_vals[rand_index]])
        rand_y = np.transpose([y_vals[rand_index]])
        sess2.run(train_step_l2, feed_dict={x_data: rand_x, y_target: rand_y})
        temp_loss_l2 = sess2.run(loss_l2, feed_dict={x_data: rand_x, y_target: rand_y})
        los_vec_l2.append(temp_loss_l2)
    sess2.close()

    plt.plot(los_vec_l1, 'k-', label = 'L1 loss')
    plt.plot(los_vec_l2, 'r--', label = 'L2 loss')
    plt.title('L1 and L2 Loss per Generation')
    plt.xlabel('Generation')
    plt.ylabel('L1 loss')
    plt.legend(loc = 'upper right')
    plt.show()

def DemingRegression():
    sess = tf.Session()
    iris = datasets.load_iris()
    x_vals = np.array([x[3] for x in iris.data])
    y_vals = np.array([y[0] for y in iris.data])
    batch_size = 50

    x_data = tf.placeholder(shape=[None, 1], dtype='float32')
    y_target = tf.placeholder(shape=[None, 1], dtype='float32')

    A = tf.Variable(tf.random_normal(shape=[1,1]))
    b = tf.Variable(tf.random_normal(shape=[1,1]))
    model_output = tf.add(tf.matmul(x_data, A), b)

    demming_numberator = tf.abs(tf.subtract(y_target, tf.add(tf.matmul(x_data, A), b)))
    demming_denominator = tf.sqrt(tf.add(tf.square(A), 1))
    loss = tf.reduce_mean(tf.truediv(demming_numberator, demming_denominator))

    init = tf.global_variables_initializer()
    sess.run(init)
    opt = tf.train.GradientDescentOptimizer(0.1)
    train_step = opt.minimize(loss)
    loss_vec = []
    for i in range(250):
        rand_index = np.random.choice(len(x_vals), size = batch_size)
        rand_x = np.transpose([x_vals[rand_index]])
        rand_y = np.transpose([y_vals[rand_index]])
        # print(rand_x.shape)
        sess.run(train_step, feed_dict = {x_data: rand_x, y_target: rand_y})
        # print(type(a))
        temp_loss = sess.run(loss, feed_dict= {x_data: rand_x, y_target: rand_y})
        loss_vec.append(temp_loss)
        if (i+1) % 50 == 0:
            print('Step # ' + str(i+1) + 'A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
            print('Loss = ' + str(temp_loss))

    [slope] = sess.run(A)
    [intercept] = sess.run(b)
    best_fit = []
    for i in x_vals:
        best_fit.append(slope * i + intercept)
    plt.plot(x_vals, y_vals, 'o', label = 'Data Points')
    plt.plot(x_vals, best_fit, 'r-', label = 'Best fit line', linewidth = 2)
    plt.legend(loc = 'upper left')
    plt.title('Sepel Length vs Pedal Width')
    plt.xlabel('Pedal Width')
    plt.ylabel('Sepal Length')
    plt.show()

def Lasso_Ridge_Regression():
    sess = tf.Session()
    iris = datasets.load_iris()
    x_vals = np.asarray([x[3] for x in iris.data])
    y_vals = np.asarray([y[0] for y in iris.data])
    batch_size = 50 
    learning_rate = 0.001
    x_data = tf.placeholder(shape = [None,1], dtype = tf.float32)
    y_data = tf.placeholder(shape = [None,1], dtype = tf.float32)

    A = tf.Variable(tf.random_normal(shape = [1,1]))
    b = tf.Variable(tf.random_normal(shape = [1,1]))

    model_output = tf.add(tf.matmul(x_data, A), b)

    lasso_param = tf.constant(0.9)
    heavyside_step = tf.truediv(1. , tf.add(1., tf.exp(tf.multiply(-100., tf.subtract(A, lasso_param)))))
    regularization_param = tf.multiply(heavyside_step, 99.)
    loss = tf.add(tf.reduce_mean(tf.square(y_data - model_output)), regularization_param)

    init = tf.global_variables_initializer()
    sess.run(init)
    opt = tf.train.GradientDescentOptimizer(0.1)
    train_step = opt.minimize(loss)
    loss_vec = []
    for i in range(1500):
        rand_index = np.random.choice(len(x_vals), size = batch_size)
        rand_x = np.transpose([x_vals[rand_index]])
        rand_y = np.transpose([y_vals[rand_index]])
        # print(rand_x.shape)
        sess.run(train_step, feed_dict = {x_data: rand_x, y_data: rand_y})
        # print(type(a))
        temp_loss = sess.run(loss, feed_dict= {x_data: rand_x, y_data: rand_y})
        loss_vec.append(temp_loss)
        if (i+1) % 50 == 0:
            print('Step # ' + str(i+1) + 'A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
            print('Loss = ' + str(temp_loss))

    [slope] = sess.run(A)
    [intercept] = sess.run(b)
    best_fit = []
    for i in x_vals:
        best_fit.append(slope * i + intercept)
    plt.plot(x_vals, y_vals, 'o', label = 'Data Points')
    plt.plot(x_vals, best_fit, 'r-', label = 'Best fit line', linewidth = 2)
    plt.legend(loc = 'upper left')
    plt.title('Sepel Length vs Pedal Width')
    plt.xlabel('Pedal Width')
    plt.ylabel('Sepal Length')
    plt.show()

def Elastic_Net_Regression():
    sess = tf.Session()
    iris = datasets.load_iris()
    x_vals = np.asarray([[x[1], x[2], x[3]] for x in iris.data])
    y_vals = np.asarray([x[0] for x in iris.data])

    batch_size = 50
    learning_rate = 0.001
    x_data = tf.placeholder(shape=[None, 3], dtype=tf.float32)
    y_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    A = tf.Variable(tf.random_normal(shape=[3, 1]))
    b = tf.Variable(tf.random_normal(shape=[1, 1]))
    model_output = tf.add(tf.matmul(x_data, A), b)

    elastic_param1 = tf.constant(1.)
    elastic_param2 = tf.constant(1.)
    l1_a_loss = tf.reduce_mean(tf.abs(A))
    l2_a_loss = tf.reduce_mean(tf.square(A))
    e1_term = tf.multiply(elastic_param1, l1_a_loss)
    e2_term = tf.multiply(elastic_param2, l2_a_loss)


    loss = tf.expand_dims(tf.add(tf.add(tf.reduce_mean(tf.square(y_data - model_output)), e1_term), e2_term), 0)
    init = tf.global_variables_initializer()
    sess.run(init)
    opt = tf.train.GradientDescentOptimizer(learning_rate)
    train_step = opt.minimize(loss)
    loss_vec = []
    for i in range(1000):
        rand_index = np.random.choice(len(x_vals), size = batch_size)
        rand_x = x_vals[rand_index]
        rand_y = np.transpose([y_vals[rand_index]])

        sess.run(train_step, feed_dict={x_data: rand_x, y_data: rand_y})
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_data: rand_y})
        loss_vec.append(temp_loss[0])
        if (i+1) % 250 == 0:
            print('Step # ' + str(i+1) + 'A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
            print('Loss = ' + str(temp_loss))

    plt.plot(loss_vec, 'k-')
    plt.title('Loss per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.show()

def normalize_cols(m):
    col_max = m.max(axis = 0)
    col_min = m.min(axis = 0)
    return (m - col_min) / (col_max - col_min)

def Logistic_Regression():
    sess = tf.Session()
    birth_file = open('./data/lowbwt.dat').read()
    birth_data = birth_file.split('\n')[2:]
    dataset = []
    for record in birth_data:
        data = [float(x) for x in record.split(' ') if x.isdigit()]
        dataset.append(data)
    dataset = np.asarray(dataset[:-3])

    x_vals = np.asarray([x[2:9] for x in dataset])
    y_vals = np.asarray([x[1] for x in dataset])

    train_indices = np.random.choice(len(x_vals), size = round(len(x_vals)*0.8), replace = False)
    test_indices = np.asarray(list(set(range(len(x_vals))) - set(train_indices)))
    x_vals_train = x_vals[train_indices]
    x_vals_test = x_vals[test_indices]
    y_vals_train = y_vals[train_indices]
    y_vals_test = y_vals[test_indices]

    x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
    x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

    batch_size = 25
    x_data = tf.placeholder(shape=[None, 7], dtype = 'float32')
    y_data = tf.placeholder(shape=[None, 1], dtype = 'float32')
    A = tf.Variable(tf.random_normal(shape=[7,1]))
    b = tf.Variable(tf.random_normal(shape=[1,1]))
    model_output = tf.add(tf.matmul(x_data, A), b)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_data, logits=model_output))
    init = tf.global_variables_initializer()
    sess.run(init)
    opt = tf.train.GradientDescentOptimizer(0.01)
    train_step = opt.minimize(loss)

    prediction = tf.round(tf.sigmoid(model_output))
    predictions_correct = tf.cast(tf.equal(prediction, y_data), tf.float32)
    accuracy = tf.reduce_mean(predictions_correct)

    loss_vec = []
    train_acc = []
    test_acc = []

    for i in range(1500):
        rand_index = np.random.choice(len(x_vals_train), size = batch_size)
        rand_x = x_vals_train[rand_index]
        rand_y = y_vals_train[rand_index].reshape(batch_size, 1)
        sess.run(train_step, feed_dict={x_data: rand_x, y_data: rand_y})
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_data: rand_y})
        loss_vec.append(temp_loss)
        temp_acc = sess.run(accuracy, feed_dict={x_data: rand_x, y_data: rand_y})
        train_acc.append(temp_acc)
        temp_acc_test = sess.run(accuracy, feed_dict={x_data: x_vals_test, y_data: np.transpose([y_vals_test])})
        test_acc.append(temp_acc_test)

    plt.plot(loss_vec, 'k-')
    plt.title('Cross Entropy Loss per Generation')
    plt.xlabel('Cross Entropu Loss')
    plt.show()
    plt.plot(train_acc, 'k-', label = 'Train Set Accuracy')
    plt.plot(test_acc, 'r--', label = 'Test Set Accuracy')
    plt.title('Train and Test Accuracy')
    plt.xlabel('Generation')
    plt.ylabel('Accuracy')
    plt.legend(loc = 'lower right')
    plt.show()

if __name__ == '__main__':
    while(True):
        print('Choose one method to conduct linear regression: ')
        print('1. Matrix Inverse Method')
        print('2. Matrix Decompose Method')
        print('3. Stochastic Gradient Descent Method')
        print('4. Compare L1 and L2 Loss Function')
        print('5. Demming Regression')
        print('6. Lasso and Ridge Regression')
        print('7. Elastic Net Regression')
        print('8. Logistic Regression')
        x = int(input())
        if (x == 1):
            Matrix_Inverse_Method()
        elif (x == 2):
            Decomposing_Method()
        elif (x == 3):
            TensorFlow_Method()
        elif (x == 4):
            Loss_Comparation()
        elif (x == 5):
            DemingRegression()
        elif (x == 6):
            Lasso_Ridge_Regression()
        elif (x == 7):
            Elastic_Net_Regression()
        elif (x == 8):
            Logistic_Regression()
        else:
            print('ERROR! Choose another method!')
