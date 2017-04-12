# Lecture 1 Welcome to TensorFlow
## Why TensorFlow?
- **Portability**: deploy computation to one or more CPUs ot GOUs in a desktop, server, or mobile device with a single API
- **Flexibility**: from Raspberry Pi, Android, Windows, iOS, Linux to server farms
- **Visualization**: TensorBoard
- **Checkpoints**: Managing experiments
- **Auto-differentiation**: autdiff

## import tensorflow as tf

1. **TF Learn**(tf.contrib.learn): simplified interface that helps users transition from the world of one-liner such as scikit-learn
2. **TF Slim**(tf.contrib.slim): lightweight library for defining, training and evaluating complex models in TensorFlow
3. High level API: Keras, TFLearn, Pretty Tensor

## Data Flow Graphs

*TensorFlow separates definition of computations from their execution*

**Phase 1**: assemble a graph
**Phase 2**: use a session to execute operations in the graph

## What's tensor

**An n-dimensional matrix**
0-d tensor: scalar(number)
1-d tensor: vector
2-d tensor: matrix
and so on

## How to get the value of node
- Create a session, assign it to variable sess so we can call it later
- Within the session, evaluate the graph to fetch the value of node

**tf.Session()**: A *Session* object encapsulates the environment in which *Operation* objects are executed, and *Tensor* objects are evaluated 

**More (sub)graphs**: Possible to break graphs into several chunks and run them parallelly across multiple CPUs, GPUs, or devices

**tf.device()**: to put part of a graph on a specific CPU or GPU

**tf.Graph()**: to add operators to a graph, set it as default

'''python
g = tf.Graph()
with g.as_default():
	x = tf.add(3,5)

sess = tf.Session(graph=g)
with tf.Session as sess:
	sess.run(x)
'''

## Why graphs
1. Save computation(only run subgraphs that lead to the values you want to fetch)
2. Break computation into small, differential pieces to facilitates auto-differentiation
3. Facilitate distributed computation, spread the work across multiple CPUs, GPUs, or devices
4. Many common machine learning models are commonly taught and visualized as directed graphs already

# Lecture 2 TensorFlow Ops

## Agenda
- Basic operation
- Tensor types
- Project speed dating
- Placeholders and feeding inputs
- Lazy loading

## TensorBoard


