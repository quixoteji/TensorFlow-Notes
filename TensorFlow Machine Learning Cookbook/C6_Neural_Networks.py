import tensorflow as tf
import numpy as np

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
        


    

if __name__ == '__main__':
    while(True):
        print('Choose one method to conduct : ')
        print('1.Operational Gates - one input')
        print('2.Operational Gates - two inputs')
        # print('3. Stochastic Gradient Descent Method')
        # print('4. Compare L1 and L2 Loss Function')
        # print('5. Demming Regression')
        # print('6. Lasso and Ridge Regression')
        # print('7. Elastic Net Regression')
        # print('8. Logistic Regression')
        x = int(input())
        if (x == 1):
            Operational_Gates1()
        elif (x == 2):
            Operational_Gates2()
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

        

