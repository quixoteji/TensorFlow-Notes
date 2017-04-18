# Best Result for different dataset

**[Discover the current state of art in objection classification](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html)**

# How tensorflow deal with MNIST

## **The Structure of MNIST in tensorflow**

- BUILD
- **__init__.py**
	- Imports mnist tutorial libraries used by tutorial examples.
- **fully_connected_feed.py**
	- Trains and Evaluates the MNIST network using a feed dictionary 
- **input_data.py**
	- Functions for downloading and reading MNIST data
- **mnist.py**
	- Implements the inference/loss/training pattern for model building
		1. inference() - Builds the model as far as is required for running the network forward to make predictions.
		2. loss() - Adds to the inference model the layers required to generate loss.
		3. training() - Adds to the loss model the Ops required to generate and apply gradients.
	- This file is used by the various "fully_connected_*.py" files and not meant to be run.
- **mnist_deep.py**
	- A deep MNIST classifier using convlolution layers.**See extensive documentation at [here](https://www.tensorflow.org/get_started/mnist/pros)**
- **mnist_softmax.py**
	- A very simple MNIST classifier.**See extensive documentation at [here](https://www.tensorflow.org/get_started/mnist/beginners)**
- **mnist_softmax_xla.py**
	- Simple MNIST classifier example with JIT XLA and timelines
- **mnist_with_summaries.py**
	- This is an unimpressive MNIST model, but it is a good example of using tf.name_scope to make a graph legible in the TensorBoard explorer, and of naming summary tags so that they are grouped meaningfully in TensorBoard. **It demonstrates the functionality of every TensorBoard dashboard.** 

## How to use MNIST in tensorflow

**from tensorflow.examples.tutorials.mnist import input_data**

mnist = input_data.read_data_sets('/data/mnist', one_hot=True)

In **input_data.py**:

**from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets**

### tf.contrib.learn Quickstart

TensorFlow’s high-level machine learning API (tf.contrib.learn) makes it easy to configure, train, and evaluate a variety of machine learning models.

tensorflow.contrib.learn.python.learn.datasets.mnist
- Functions for downloading and reading MNIST data.
