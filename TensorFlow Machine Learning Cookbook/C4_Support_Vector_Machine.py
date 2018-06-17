import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

def Linear_SVM():
    sess = tf.Session()

    iris = datasets.load_iris()
    x_vals = np.asarray([[x[0], x[3]] for x in iris.data])
    y_vals = np.asarray([1 if y == 0 else -1 for y in iris.target])

    train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
    test_indices = np.asarray(list(set(range(len(x_vals)))-set(train_indices)))
    x_vals_train = x_vals[train_indices]
    y_vals_train = y_vals[train_indices]
    x_vals_test = x_vals[test_indices]
    y_vals_test = np.transpose([y_vals[test_indices]])

    batch_size = 100
    x_data = tf.placeholder(shape = [None, 2], dtype = 'float32')
    y_data = tf.placeholder(shape = [None, 1], dtype = 'float32') 

    A = tf.Variable(tf.random_uniform(shape = [2,1]))
    b = tf.Variable(tf.random_uniform(shape = [1,1]))

    model_output = tf.subtract(tf.matmul(x_data, A), b)

    l2_norm = tf.reduce_sum(tf.square(A))

    alpha = tf.constant([0.1])

    classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(model_output, y_data))))

    loss = tf.add(classification_term, tf.multiply(alpha, l2_norm))

    prediction = tf.sign(model_output)

    acc = tf.reduce_mean(tf.cast(tf.equal(prediction, y_data), tf.float32))

    opt = tf.train.GradientDescentOptimizer(0.01)
    train_step = opt.minimize(loss)

    init = tf.global_variables_initializer()
    sess.run(init)

    loss_vec = []
    train_acc = []
    test_acc = []
    for i in range(1000):
        rand_index = np.random.choice(len(x_vals_train), size = batch_size)
        rand_x = x_vals_train[rand_index]
        rand_y = np.transpose([y_vals_train[rand_index]])

        sess.run(train_step, feed_dict = {x_data: rand_x, y_data: rand_y})

        temp_loss = sess.run(loss, feed_dict = {x_data: rand_x, y_data: rand_y})
        loss_vec.append(temp_loss)
        temp_train_acc = sess.run(acc, feed_dict = {x_data: rand_x, y_data: rand_y})
        train_acc.append(temp_train_acc)
        temp_test_acc = sess.run(acc, feed_dict = {x_data: x_vals_test, y_data: y_vals_test})
        test_acc.append(temp_test_acc)

        if (i+1) % 100 == 0:
            print('Step #' + str(i+1) + 'A = ' + str(sess.run(A)) + ' b = ' +str(sess.run(b)))
            print('Loss = ' + str(temp_loss))
            print('Train Accuracy = ' + str(temp_train_acc))
            print('Test Accuracy = ' + str(temp_test_acc))

    [[a1], [a2]] = sess.run(A)
    [[b]] = sess.run(b)
    slop = -a2/a1
    intercept = b/a1
    x1_vals = [d[1] for d in x_vals]
    best_fit = []
    for i in x1_vals:
        best_fit.append(slop*i + intercept)
    setosa_x = [d[1] for i, d in enumerate(x_vals) if y_vals[i] == 1]
    setosa_y = [d[0] for i, d in enumerate(x_vals) if y_vals[i] == 1]

    not_setosa_x = [d[1] for i, d in enumerate(x_vals) if y_vals[i] == -1]
    not_setosa_y = [d[0] for i, d in enumerate(x_vals) if y_vals[i] == -1]

    plt.plot(setosa_x, setosa_y, 'o', label = 'I.setosa')
    plt.plot(not_setosa_x, not_setosa_y, 'x', label = 'Non-setosa')
    plt.plot(x1_vals, best_fit, 'r-', label = 'Linear Seperator', linewidth = 3)
    plt.ylim([0,10])
    plt.legend(loc='lower right')
    plt.title('Sepa Length vs Pedal Width')
    plt.xlabel('Pedal Width')
    plt.ylabel('Sepal Length')
    plt.show()

    plt.plot(train_acc, 'k-', label = "Training Accuracy")
    plt.plot(test_acc, 'r--', label = 'Test Accuracy')
    plt.title("Train and Test Set Accuracies")
    plt.xlabel('Generation')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

    plt.plot(loss_vec, 'k-')
    plt.ylim([0, 2.5])
    plt.title('Loss per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.show()

def LinearRegressionSVM():
    sess = tf.Session()
    iris = datasets.load_iris()
    x_vals = np.array([x[3] for x in iris.data])
    y_vals = np.array([x[0] for x in iris.data])
    train_indices = np.random.choice(len(x_vals), size = round(len(x_vals)*0.8), replace=False)
    test_indices = np.asarray(list(set(range(len(x_vals))) - set(train_indices)))
    x_vals_train = x_vals[train_indices]
    y_vals_train = y_vals[train_indices]
    x_vals_test = x_vals[test_indices]
    y_vals_test = y_vals[test_indices]

    A = tf.Variable(tf.random_normal(shape = [1,1]))
    b = tf.Variable(tf.random_normal(shape = [1,1]))
    x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    y_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    model_output = tf.add(tf.matmul(x_data, A), b)

    epsilon = tf.constant([0.5])
    loss = tf.reduce_mean(tf.maximum(0., tf.subtract(tf.abs(tf.subtract(model_output, y_data)),epsilon)))
    opt = tf.train.GradientDescentOptimizer(0.075) 
    train_step = opt.minimize(loss)

    init = tf.global_variables_initializer()
    sess.run(init)

    train_loss = []
    test_loss = []
    for i in range(200):
        rand_index = np.random.choice(len(x_vals_train), size = 50)
        rand_x = np.transpose([x_vals_train[rand_index]])
        rand_y = np.transpose([y_vals_train[rand_index]])
        test_x = np.transpose([x_vals_test])
        test_y = np.transpose([y_vals_test])
        sess.run(train_step, feed_dict={x_data:rand_x, y_data:rand_y})
        temp_train_loss = sess.run(loss, feed_dict={x_data:rand_x, y_data:rand_y})
        train_loss.append(temp_train_loss)
        temp_test_loss = sess.run(loss, feed_dict={x_data:test_x, y_data:test_y})
        test_loss.append(temp_test_loss)
        if (i+1) % 50 == 0:
            print('-----------------------')       
            print('Generation: ' + str(i+1))
            print('A = ' + str(sess.run(A)) +  'b = ' + str(sess.run(b)))
            print('Train Loss: ' + str(temp_train_loss))
            print('Test Loss: ' + str(temp_test_loss))        
    [[slope]] = sess.run(A)
    [[intercept]] = sess.run(b)
    [width] = sess.run(epsilon)
    best_fit = []
    best_fit_upper = []
    best_fit_lower = []
    for i in x_vals:
        best_fit.append(slope*i+intercept)
        best_fit_upper.append(slope*i + intercept+width)
        best_fit_lower.append(slope*i + intercept-width)

    plt.plot(x_vals, y_vals, 'o', label = 'Data Points')
    plt.plot(x_vals, best_fit, 'r-', label = 'SVM Regression Line', linewidth = 3)
    plt.plot(x_vals, best_fit_upper, 'r--', linewidth = 2)
    plt.plot(x_vals, best_fit_lower, 'r--', linewidth = 2)
    plt.plot([0, 10])
    plt.legend(loc='lower right')
    plt.title('Sepal Length vs Pedal Width')
    plt.xlabel('Pedal Width')
    plt.ylabel('Sepal Length')
    plt.show()
    plt.plot(train_loss, 'k--', label = 'Train Set Loss')
    plt.plot(test_loss, 'r--', label = 'Test Set Loss')
    plt.title('L2 Loss per Generation')
    plt.xlabel('Generation')
    plt.ylabel('L2 Loss')
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    while(True):
        print('Choose one method to conduct support vector machines: ')
        print('1. Linear SVM')
        print('2. Reduction to Linear Regression')
        print('3. SVM with Kernals')
        # print('4. Compare L1 and L2 Loss Function')
        # print('5. Demming Regression')
        # print('6. Lasso and Ridge Regression')
        # print('7. Elastic Net Regression')
        # print('8. Logistic Regression')
        x = int(input())
        if (x == 1):
            Linear_SVM()
        elif (x == 2):
            LinearRegressionSVM()
        # elif (x == 3):
        #     TensorFlow_Method()
        # elif (x == 4):
        #     Loss_Comparation()
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

    





