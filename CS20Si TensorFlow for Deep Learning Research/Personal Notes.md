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
- Placeholders and feeding inputs
- Lazy loading

## TensorBoard
### Codes
writer = tf.summary.FileWriter("./graphs", sess.graph)

with tf.Session() as sess:
### Run it
$ python [program].py

$ tensorboard --logdir="./graphs" --port 6006

### Open browser and go to: http://localhost:6006/

## tf.constant
tf.constant(value, dtype=None, shape=None, name='Const', verify_shape=False)

tf.zeros(shape, dtype=tf.float32, name=None)
*creates a tensor of shape and all elements will be zeros(when ran in session)*

tf.zeros_like(input_tensor, dtype=None, name=None, optimize=True)
*creates a tensor of shape and type(unless type is specified) as the input_tensor but all elements are zeros*

tf.ones(shape, dtype=tf.float32, name=None)
tf.ones_like(input_tensor, dtype=None, name=None, optimize=True)

tf.fill(dims, value, name=None)
*creats a tensor filled with a scalar value*

### Constant as sequences
tf.linspace(start, stop, num, name=None)
tf.range(start, limit=None, delta=1, dtype=None, name='range')
**Tensor objects are not iterable**

### Randomly Generated Constants
tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)

tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)

tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)

tf.random_shuffle(value, seed=None, name=None)

tf.random_crop(value, size, seed=None, name=None)

tf.multinomial(logits, num_samples, seed=None, name=None)

tf.random_gamma(shape, alpha, beta=None, dtype=tf.float32, seed=None, name=None)

**tf.set_random_seed(seed)**

## Operations

tf.add()

tf.add_n()

tf.mul()

tf.matmul()

tf.div()

tf.mod()

##TensorFlow Data Types

- **TensorFlow integrates seamlessly with NumPy**
- **Can pass numpy types to TensorFlow ops**
- **For tf.Session.run(fetches)**: if the requested fetch is a Tensor, then the output of will be a NumPy ndarray
- **Do not use Python native types for tensors because TensorFlow has to infer Python type**
- **Beware when using NumPy arrays because NumPy and TensorFlow might become not so compatible in the future**
- **Do not use constants, constants are stored in the graph definition**:
***This makes loading graphs expensive when constants are big, only use constants for primitive types. Use variables or readers for more data that requires more memory.***

## Variables
**tf.Variables()**

- The easiest way is initializing all variables at once:

>init = tf.global_variables_initializer()
>
>with tf.Session() as sess:
>
>sess.run(init)

- Initialize only a subset of variables:

>init_ab = tf.variable_initializer([a, b], name="init_ab")
>
>with tf.Session as sess:
>
>sess.run(init_ab)

- Initialize a single variable

>W = tf.Variable(tf.zeros[784, 10])
>
>with tf.Session as sess:
>
>sess.run(W.initializer)

#### Eval()
>W = tf.Variable(tf.truncated_normal([700, 10]))
>
>with tf.Session() as sess:
>
>sess.run(W.initializer)
>
>print(W.eval())

#### tf.Variable.assign()
>W = tf.Variable(10)
>
>W.assign(100)
>
>with tf.Session() as sess:
>
>sess.run(W.initializer)
>
>print(W.eval()) # >> 10 (W.assign(100) doesn't assign the value 100 to W. It creates an assign op, and that op needs to be run to take effect)

---

>W = tf.Variable(10)
>
>W.assign(100)
>
>with tf.Session() as sess:
>
>sess.run(assign_op)
>
>print(W.eval()) # >> 100 (You don't need to initialize variable because assign_op does it for you)

## Session VS InteractiveSession
*You sometimes see InteractiveSession instead of Session. The only difference is an InteractiveSession makes itself the default*

## Control Dependencies

**tf.Graph.control_dependencies(control_input)**: define which ops should be run first

## Placehoders

A TF program often has 2 phase:
1. Assemble a graph
2. Use a session to execute operations in the graph

*why placeholders*
We can later supply their own data when they need to execute the computation

*tf.placeholder(dtype, shape=None, name=None)*

### Feed the values to placeholders using a dictionary

### Placeholders are valid ops

### You can feed_dict any feedable tensor. Placeholder is just a way to indicate that something must be fed

### tf.Graph.is_feedable(tensor)

True if and only if tensor is feedable

## Lazy Loading

***Defer creating/initiablizing an object until it is needed***

### Graph description
**tf.get_default_graph().as_graph_def()**

# Lecture 3 Basic Models in TensorFlow

## Agenda
- Review
- Linear regression in TensorFlow
- Optimizers
- Logistic regression on MNIST
- Loss function

## Linear Regression

- Phase 1: Assemble our graph
	- Step 1: Read in data
	- Step 2: Create placeholders for input and labels (X,Y)
	- Step 3: Create weight and bias (w, b)
	- Step 4: Build model to predict Y
	- Step 5: Specify loss function
	- Step 6: Create optimizer
- Phase 2: Train our model
	- Initialize variables
	- Run optimizer op(with data fed into placeholders for inputs and labels)
- See model in TensorBoard
	- Step 1: writer = tf.summary.FileWriter('./graph', sess.graph)
	- Step 2: $ tensorboard --logdir='./graph'


## Optimizer

*Session looks at all **trainable** variables that optimizer depends on and update them*

### List of optimizers in TF

- tf.train.GradientDescentOptimizers
- tf.train.AdagradOptimizer
- tf.train.MomentumOptimizer
- tf.train.AdamOptimizer
- tf.train.ProximalGradientDescentOptimizer
- tf.train.ProximalAdagradOptimizer
- tf.train.RMSPropOptimizer
- And more

## Huber loss

Robust to outliers

Intuition: if the difference between the predicted value and the real value is small, square it

If it's large, take its absolute value

# Lecture 4 Structure your model

## Agenda
- Overall structure of a model in TensorFlow
- word2vec
- Name scope
- Embedding visualization

### Phrase 1: Assemble graph
1. Define placeholders for input and output
2. Define the weights
3. Define the inference model
4. Define loss function
5. Define optimizer

### Phrase 2: Compute

## Word Embedding
*Capture the semantic relationships between words*

### Embedding Lookup
tf.nn.embedding_lookup(params, ids, partition_strategy='mod', name=None, validate_indices=True, max_norm=None)

### NCE Loss
tf.nn.nce_loss(weights, biases, labels, inputs, num_sampled, num_classes, ...)

## Name scope
*Group nodes together*

with tf.name_scope(name)

# Lecture 5 Manage Experiments

## Agenda
- More word2vec
- tf.train.Saver
- tf.summary
- Randomization
- Data Readers

**tf.gradients(y, [xs])**:
Take derivative of y with respect to each tensor in the list [xs]

## Manage experiments

###**tf.train.Saver**:
saves graph's variables in binary files

#### Saves sessions, not graphs
tf.train.Saver.save(sess, save_path, global_step=None, ...)
**Only save variables, not graph**
**Checkpoints map variable names to tensors**

#### Save parameters after 1000 steps

>_#define model
>
>_#create a saver object
>
>saver = tf.train.Saver()
>
>_#launch a session to compute the graph
>
>with tf.Session as sess
>
>_#actual training loop:
>
>for step in range(training_steps):
>
>sess.run([optimizer])
>
>if (step+1) % 1000==0:
>
>saver.save(sess, 'checkpoints_directory/model_name', global_step=model.global_step)

#### Each saved step is a checkpoint
**Global step**
*Common in TensorFlow*
self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
*Need to tell optimizer to increment global step*
self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)

#### Restore variables
saver.restore(sess, 'checkpoints/name_of_the_checkpoints')

e.g. saver.restore(sess, 'checkpoints/skip-gram-99999')

#### Restore the latest checkpoint

ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))

if ckpt and ckpt.model_checkpoint_path:
	saver.restore(sess, ckpt.model_checkpoint_path)
	
- *checkpoint keeps track of the latest checkpoint*
- *Safeguard to restore checkpoints only when there are checkpoints*

### **tf.summary**

*Visualize our summary statistics during our training*

- tf.summary.scalar
- tf.summary.histogram
- tf.summary.image

#### Step 1: create summaries

>with tf.name_scope("summaries"):
>	
>__	tf.summary.scalar("loss", self.loss)
>
>__tf.summary.scalar("accuracy", self.accuracy)
>
>__tf.summary.histogram("histogram loss", self.loss)
>
>_# merge them all
>
>__self.summary_op = tf.summary.merge_all()

#### Step 2: rum them

>loss_batch, _, summary = sess.run([model.loss, model.optimizer, model.summary_op], feed_dict=feed_dict)

*Like everything else in TF, summaries are ops*

#### Step 3: write summaries to file

>writer.add_summary(summary, global_step=step)

### Control Randomization

#### Op level random seed
>my_var = tf.Variable(tf.truncated_normal((-1.0, 1.0),stddev=0.1, seed=0))

#### Sessions keep track of random state
#### Graph level seed
>tf.set_random_seed(seed)

## Data Readers
### Problem with feed_dict
- Slow when client and workers are on different machines
- Readers allow us to load data directly into the worker process

### Different Readers for different file types

**tf.TextLineReader**: Outputs the lines of a file delimited by newlines(text files, CSV files)

**tf.FixedLengthRecordReader**: Outputs the entire file when all files have same fixed lengths(each MNIST has 28 * 28 pixels, CIFAR-10 32 * 32 * 3)

**tf.WholeFileReader**:
Outputs the entire file content

**tf.TFRecordReader**:
Reads samples from TensorFlow's own binary format(TFRecord)

**tf.ReaderBase**:
To allow you to create your own readers

### Read in files from queues
>filename_queue = tf.train.string_input_producer(["file0.csv", "file1.csv"])
>
>reader = tf.TextLineReader()
>
>key, value = reader.read(filename_queue)

### Threads & Queues
**You can use tf.Coordinator and tf.QueueRunner to manage your queues**

>with tf.Session() as sess:
>
>__#start populating the filename queue
>
>__coord = tf.train.Coordinator()
>
>__threads = tf.train.start_queue_runner(coord=coord)



