# How to transfer our data into Tensorflow TFRecord
- **Transform into TFRecord**
- **Read and decode**
- **Generate batches**
- **Feed in Tensorflow computational graph**
- **Train, validation, test**

## TFRecord

- Standord Tensorflow format
- What we need:
	1. **tf.python_io.TFRecordWriter**: to write records to a TFRecords file
		- **path**: the path to the TFRecords file.
		- **write(record)**: write a record to the file
	2. **tf.TFRecordReader**: a Reader that outputs the records from a TFRecords file
		- **read(queue)**: dequeue a work unit from queue and return (key, value) pair.
	3. **tf.parse_single_example**: to parse a single Example.

## Dataset: notMNIST

- *notMNIST* contains *letters A-J(10 classes) *taken from different fonts
- Train dataset:
	- 530K 28 * 28 gray scale images
	- 53K images for each class
- Test dataset:
	- 18K
	- 1.8K for each class
	
### Notes
- Bad bugs exist in notMNIST
- Use(try: except:) to read all the images 

 	  	 	