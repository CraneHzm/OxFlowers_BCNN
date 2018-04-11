# Copyright (c) 2018/4/11 Hu Zhiming JimmyHu@pku.edu.cn All Rights Reserved.
# The Bilinear CNN estimator for 17_OxFlowers Dataset.


#######################################
# Network Architecture: 
# input layer
# conv3_32
# maxpool
# conv3_64
# maxpool
# conv3_128
# maxpool
# conv3_256
# maxpool
# conv3_512
# maxpool
# conv3_512
# Bilinear Layer(The 2 convolution layers which are combined by the Bilinear Layer are identical, you can also combine 2 different convolution layers)
# output layer(logits Layer)
#######################################


#######################################
# Usage: 
# The model has been trained for 4000 global steps and the accuracy on test set is 85%.
# Run this code to test the accuracy of our model.
# If you want to retrain the model, uncomment the training code in the main function and set the 'Steps' & 'Loops' global variables to control the training process.
#######################################



# import the libs(libs & future libs).
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import datetime


# the params of the input dataset.
# the width, height & channels of the images in the input TFRecords file.
ImageWidth = 224
ImageHeight = 224
ImageChannels = 3
# the number of categories.
CategoryNum = 17

# the params for training.
# the batch size for training.
Batch_Size = 85
# Steps is the number of training steps in a time. 
Steps = 200
# Loops is the total number of training times.
# Global_Training_Steps = Steps* Loops
Loops = 0


# Output thogging info.
# Use tensorboard --logdir=PATH to view the graphs.
tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
	"""the model function for CNN OxFlower."""
	# Input Layer
	# Reshape X to 4-D tensor: [batch_size, width, height, channels]
	# oxflower images are 224*224 pixels, and have 3 color channels
	input_layer = tf.reshape(features['x'], [-1, ImageWidth, ImageHeight, ImageChannels])
	# print(type(features))
	# print(features)
	# print(input_layer.shape)
	# print(type(labels))
	# input('1:')
	
	# Convolutional Layer #1
	conv1 = tf.layers.conv2d(
		inputs = input_layer,
		filters = 32,
		kernel_size = 3,
		padding = "same")

	# Batch Normalization Layer #3
	bn1 = tf.layers.batch_normalization(inputs = conv1)
	layer1 = tf.nn.relu(bn1)

	
	# Pooling Layer #1
	pool1 = tf.layers.max_pooling2d(inputs = layer1, pool_size = [2, 2], strides = 2)

	# Convolutional Layer #2
	conv2 = tf.layers.conv2d(
		inputs = pool1,
		filters = 64,
		kernel_size = 3,
		padding = "same")

	# Batch Normalization Layer #3
	bn2 = tf.layers.batch_normalization(inputs = conv2)
	layer2 = tf.nn.relu(bn2)
	
	# Pooling Layer #2
	pool2 = tf.layers.max_pooling2d(inputs = layer2, pool_size = [2, 2], strides = 2)

	# Convolutional Layer #3
	conv3 = tf.layers.conv2d(
		inputs = pool2,
		filters = 128,
		kernel_size = 3,
		padding = "same")
	
	# Batch Normalization Layer #3
	bn3 = tf.layers.batch_normalization(inputs = conv3)
	layer3 = tf.nn.relu(bn3)
	
	# Pooling Layer #3
	pool3 = tf.layers.max_pooling2d(inputs = layer3, pool_size = [2, 2], strides = 2)
	
	# Convolutional Layer #4
	conv4 = tf.layers.conv2d(
		inputs = pool3,
		filters = 256,
		kernel_size = 3,
		padding = "same")

	# Batch Normalization Layer #4
	bn4 = tf.layers.batch_normalization(inputs = conv4)
	layer4 = tf.nn.relu(bn4)
	
	# Pooling Layer #4
	pool4 = tf.layers.max_pooling2d(inputs = layer4, pool_size = [2, 2], strides = 2)	

	# Convolutional Layer #5
	conv5 = tf.layers.conv2d(
		inputs = pool4,
		filters = 512,
		kernel_size = 3,
		padding = "same")	

	# Batch Normalization Layer #5
	bn5 = tf.layers.batch_normalization(inputs = conv5) 
	layer5 = tf.nn.relu(bn5)
	pool5 = tf.layers.max_pooling2d(inputs = layer5, pool_size = [2, 2], strides = 2)	

	# Convolutional Layer #6
	conv6 = tf.layers.conv2d(
		inputs = pool5,
		filters = 512,
		kernel_size = 3,
		padding = "same",)	
	
	# Batch Normalization Layer #6
	bn6 = tf.layers.batch_normalization(inputs = conv6)
	layer6 = tf.nn.relu(bn6)
	
	# print(layer6)
	# input("input: ")
	
	# The bilinear layer.
	# We combine the 2 identical convolution layers in our code. You can also combine 2 different convolution layers.
	# The bilinear layer is connected to the final output layer(the logits layer).
	phi_I = tf.einsum('ijkm,ijkn->imn', layer6, layer6)
	# print(phi_I)
	phi_I = tf.reshape(phi_I, [-1,512*512])
	# print(phi_I)
	phi_I = tf.divide(phi_I, 49)
	# print(phi_I)
	phi_I = tf.layers.batch_normalization(inputs = phi_I)
	# print(phi_I)
	# input("input: ")

	# Add dropout operation; 0.2 probability that element will be kept
	dropout_1 = tf.layers.dropout(
		inputs = phi_I, rate = 0.8, training=mode == tf.estimator.ModeKeys.TRAIN)

	# Logits layer
	# Output Tensor Shape: [batch_size, CategoryNum]
	# Default: activation=None, maintaining a linear activation.
	logits = tf.layers.dense(inputs = dropout_1, units = CategoryNum)

	predictions = {
		# Generate predictions (for PREDICT and EVAL mode)
		"classes": tf.argmax(input=logits, axis=1),
		# Add `softmax_tensor` to the graph. It is used for PREDICT and by the
		# `logging_hook`.
		"probabilities": tf.nn.softmax(logits, name="softmax_tensor")}
	
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	# Calculate Loss (for both TRAIN and EVAL modes)
	# No need to use one-hot labels.
	loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

	# Calculate evaluation metrics.
	accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"], name='acc_op')
	eval_metric_ops = {'accuracy': accuracy}
	# Use tensorboard --logdir=PATH to view the graphs.
	# The tf.summary.scalar will make accuracy available to TensorBoard in both TRAIN and EVAL modes. 
	tf.summary.scalar('accuracy', accuracy[1])
	
	# Configure the Training Op (for TRAIN mode)
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
		train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	# Add evaluation metrics (for EVAL mode)
	if mode == tf.estimator.ModeKeys.EVAL:
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)  
	
	
def read_and_decode(filename):
	"""
	read and decode a TFRecords file.
	returns numpy array objects.
	pipeline: TFRecords --> queue --> serialized_example --> dict.
	"""
	# Output strings (e.g. filenames) to a queue for an input pipeline.
	filename_queue = tf.train.string_input_producer([filename])
	# print(filename_queue)
	# A Reader that outputs the records from a TFRecords file.
	reader = tf.TFRecordReader()
	# reader.read(queue)
	# Args queue: A Queue or a mutable string Tensor representing a handle to a Queue, with string work items.
	# Returns: A tuple of Tensors (key, value). key: A string scalar Tensor. value: A string scalar Tensor.
	_, serialized_example = reader.read(filename_queue)
	# print(serialized_example)
	
	# Parses a single Example proto.
	# Returns a dict mapping feature keys to Tensor and SparseTensor values.
	features = tf.parse_single_example(serialized_example,features={
	'label': tf.FixedLenFeature([], tf.int64), 'img_raw' : tf.FixedLenFeature([], tf.string),})
	# Reinterpret the bytes of a string as a vector of numbers.
	imgs = tf.decode_raw(features['img_raw'], tf.uint8)
	# print(img.dtype)
	# print(img.shape)
	# Reshapes a tensor.
	imgs = tf.reshape(imgs, [-1, ImageWidth, ImageHeight, ImageChannels])  
	# cast the data from (0, 255) to (-0.5, 0.5)
	# (-0.5, 0.5) may be better than (0, 1).
	imgs = tf.cast(imgs, tf.float32) * (1. / 255) - 0.5
	labels = tf.cast(features['label'], tf.int64)  
	
	# print(type(imgs))
	# print(imgs.shape)
	# print(type(labels))
	# print(labels.shape)
	return imgs, labels  


def parse_function(example_proto):
	"""parse function is used to parse a single TFRecord example in the dataset."""
	# Parses a single Example proto.
	# Returns a dict mapping feature keys to Tensor and SparseTensor values.
	features = tf.parse_single_example(example_proto,features={
	'label': tf.FixedLenFeature([], tf.int64), 'img_raw' : tf.FixedLenFeature([], tf.string),})
	# Reinterpret the bytes of a string as a vector of numbers.
	imgs = tf.decode_raw(features['img_raw'], tf.uint8)
	# Reshapes a tensor.
	imgs = tf.reshape(imgs, [ImageWidth, ImageHeight, 3])  
	# cast the data from (0, 255) to (-0.5, 0.5)
	# (-0.5, 0.5) may be better than (0, 1).
	imgs = tf.cast(imgs, tf.float32) * (1. / 255) - 0.5
	labels = tf.cast(features['label'], tf.int64) 
	return {'x': imgs}, labels


def train_input_fn(tfrecords, batch_size):
	"""
	An input function for training mode.
	tfrecords: the filename of the training TFRecord file, batch_size: the batch size.
	"""
	# read the TFRecord file into a dataset.
	dataset = tf.data.TFRecordDataset(tfrecords)
	# parse the dataset.
	dataset = dataset.map(parse_function)
	# the size of the buffer for shuffling.
	# buffer_size should be greater than the number of examples in the Dataset, ensuring that the data is completely shuffled. 
	buffer_size = 10000
	# Shuffle, repeat, and batch the examples.
	dataset = dataset.shuffle(buffer_size).repeat().batch(batch_size)
	# print(dataset)
	
	# make an one shot iterator to get the data of a batch.
	train_iterator = dataset.make_one_shot_iterator()
	# get the features and labels.
	features, labels = train_iterator.get_next()
	# print(features)
	# print(labels)
	return features, labels

def eval_input_fn(tfrecords, batch_size):
	"""
	An input function for evaluation mode.
	tfrecords: the filename of the evaluation/test TFRecord file, batch_size: the batch size.
	"""
	# read the TFRecord file into a dataset.
	dataset = tf.data.TFRecordDataset(tfrecords)
	# parse the dataset.
	dataset = dataset.map(parse_function)
	
	# Shuffle, repeat, and batch the examples.
	dataset = dataset.batch(batch_size)
	# print(dataset)
		# make an one shot iterator to get the data of a batch.
	eval_iterator = dataset.make_one_shot_iterator()
	# get the features and labels.
	features, labels = eval_iterator.get_next()
	# print(features)
	# print(labels)
	return features, labels


def main(unused_argv):

	# Create the Estimator
	oxflower_classifier = tf.estimator.Estimator(
		model_fn=cnn_model_fn, 
		model_dir="Models/BCNN_85/")
	
	"""
	# Uncomment this to retain the model.
	# train and validate the model in a loop.
	# the start time of training.
	start_time = datetime.datetime.now()
	for i in range(Loops):
		# Train the model
		oxflower_classifier.train(
			input_fn=lambda:train_input_fn('Dataset/train_aug.tfrecords', Batch_Size),
			steps = Steps)
		# Evaluate the model on validation set.
		eval_results = oxflower_classifier.evaluate(input_fn=lambda:eval_input_fn('Dataset/validation.tfrecords', Batch_Size))
		# Calculate the accuracy of our CNN model.
		accuracy = eval_results['accuracy']*100
		print('\n\ntraining steps: {}'.format((i+1)*Steps))
		print('Validation set accuracy: {:0.2f}%\n\n'.format(accuracy))
	
	# the end time of training.
	end_time = datetime.datetime.now()
	print('\n\n\ntraining starts at: {}'.format(start_time))
	print('\ntraining ends at: {}\n\n\n'.format(end_time))
	"""
	
	
	# Evaluate the model on test set.
	eval_results = oxflower_classifier.evaluate(input_fn=lambda:eval_input_fn('Dataset/test.tfrecords', Batch_Size))
	# Calculate the accuracy of our CNN model.
	accuracy = eval_results['accuracy']*100
	print('\n\nTest set accuracy: {:0.2f}%\n\n'.format(accuracy))	
	
if __name__ == "__main__":
	"""tf.app.run() runs the main function in this module by default."""
	tf.app.run()
