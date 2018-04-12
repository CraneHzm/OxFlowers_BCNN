# Copyright (c) 2018/4/8 Hu Zhiming JimmyHu@pku.edu.cn All Rights Reserved.
# Create the tarin & test dataset from the given train & test images.

# import the libs(libs & future libs).
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os  
import tensorflow as tf  
# Image is used to process images.
from PIL import Image 

# Get the current work directory.
cwd = os.getcwd()
# create a list of the files in the given directory.
train_dir = os.listdir(cwd+"/Train/")
validation_dir = os.listdir(cwd+"/Validation/")
test_dir = os.listdir(cwd+"/Test/")
# a class to write records to a TFRecords file.
# train_writer = tf.python_io.TFRecordWriter("../Dataset/train.tfrecords")  
validation_writer = tf.python_io.TFRecordWriter("../Dataset/validation.tfrecords") 
test_writer = tf.python_io.TFRecordWriter("../Dataset/test.tfrecords") 

# the width & height to resize the images.
ImageWidth = 224
ImageHeight= 224

# the number of images in a category.
Number = 80

"""
# create the train tfrecords.
for name in train_dir:  
	# the path of an image.
	image_path = cwd + "/Train/"+name  
	# read the image.
	img = Image.open(image_path)  
	# resize the image.
	img = img.resize((ImageWidth, ImageHeight))  
	# convert the image to bytes(raw data).
	img_raw = img.tobytes()
	# calculate the label of this image.
	num = name.split('.')
	# num is the id of the image in the original image dataset.
	num = int(num[0])
	label = int((num-1)/Number)
	# print('num: {}, Number: {}, label: {}'.format(num, Number, label))
	# example defines the data format.
	example = tf.train.Example(features=tf.train.Features(feature={  
	"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),  
	'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))  
	}))  
	
	# Serialize the data to string and write it.
	train_writer.write(example.SerializeToString())
	
	print('Image Name: {}, label: {}'.format(name, label))  

# close the TFRecords writer.	
train_writer.close()  
"""

# create the validation tfrecords.
for name in validation_dir:  
	# the path of an image.
	image_path = cwd + "/Validation/"+name  
	# read the image.
	img = Image.open(image_path)  
	# resize the image.
	img = img.resize((ImageWidth, ImageHeight))  
	# convert the image to bytes(raw data).
	img_raw = img.tobytes()
	# calculate the label of this image.
	num = name.split('.')
	# num is the id of the image in the original image dataset.
	num = int(num[0])
	label = int((num-1)/Number)
	# print('num: {}, Number: {}, label: {}'.format(num, Number, label))
	# example defines the data format.
	example = tf.train.Example(features=tf.train.Features(feature={  
	"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),  
	'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))  
	}))  
	
	# Serialize the data to string and write it.
	validation_writer.write(example.SerializeToString())
	
	print('Image Name: {}, label: {}'.format(name, label))  

# close the TFRecords writer.	
validation_writer.close()

# create the test tfrecords.
for name in test_dir:  
	# the path of an image.
	image_path = cwd + "/Test/"+name  
	# read the image.
	img = Image.open(image_path)  
	# resize the image.
	img = img.resize((ImageWidth, ImageHeight))  
	# convert the image to bytes(raw data).
	img_raw = img.tobytes()
	# calculate the label of this image.
	num = name.split('.')
	# num is the id of the image in the original image dataset.
	num = int(num[0])
	label = int((num-1)/Number)
	# print('num: {}, Number: {}, label: {}'.format(num, Number, label))
	# example defines the data format.
	example = tf.train.Example(features=tf.train.Features(feature={  
	"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),  
	'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))  
	}))  
	
	# Serialize the data to string and write it.
	test_writer.write(example.SerializeToString())
	
	print('Image Name: {}, label: {}'.format(name, label))  

# close the TFRecords writer.	
test_writer.close()

# the augmented train set.
# a class to write records to a TFRecords file.
train_aug_writer = tf.python_io.TFRecordWriter("../Dataset/train_aug.tfrecords")
trainFlip_dir = os.listdir(cwd+"/TrainFlip/")
trainCrop_dir = os.listdir(cwd+"/TrainCrop/")
trainNoise_dir = os.listdir(cwd+"/TrainNoise/")

# write the original train data.
for name in train_dir:  
	# the path of an image.
	image_path = cwd + "/Train/"+name  
	# read the image.
	img = Image.open(image_path)  
	# resize the image.
	img = img.resize((ImageWidth, ImageHeight))  
	# convert the image to bytes(raw data).
	img_raw = img.tobytes()
	# calculate the label of this image.
	num = name.split('.')
	# num is the id of the image in the original image dataset.
	num = int(num[0])
	label = int((num-1)/Number)
	# print('num: {}, Number: {}, label: {}'.format(num, Number, label))
	# example defines the data format.
	example = tf.train.Example(features=tf.train.Features(feature={  
	"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),  
	'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))  
	}))  
	
	# Serialize the data to string and write it.
	train_aug_writer.write(example.SerializeToString())
	print('Image Name: {}, label: {}'.format(name, label))  

# write the flipped train data.
for name in trainFlip_dir:  
	# the path of an image.
	image_path = cwd + "/TrainFlip/"+name
	# read the image.
	img = Image.open(image_path)  
	# resize the image.
	img = img.resize((ImageWidth, ImageHeight))  
	# convert the image to bytes(raw data).
	img_raw = img.tobytes()
	# calculate the label of this image.
	num = name.split('.')
	# num is the id of the image in the original image dataset.
	num = int(num[0])
	label = int((num-1)/Number)
	# print('num: {}, Number: {}, label: {}'.format(num, Number, label))
	# example defines the data format.
	example = tf.train.Example(features=tf.train.Features(feature={  
	"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),  
	'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))  
	}))  
	
	# Serialize the data to string and write it.
	train_aug_writer.write(example.SerializeToString())
	print('Image Name: {}, label: {}'.format(name, label)) 

# write the cropped train data.
for name in trainCrop_dir:  
	# the path of an image.
	image_path = cwd + "/TrainCrop/"+name
	# read the image.
	img = Image.open(image_path)  
	# resize the image.
	img = img.resize((ImageWidth, ImageHeight))  
	# convert the image to bytes(raw data).
	img_raw = img.tobytes()
	# calculate the label of this image.
	num = name.split('.')
	# num is the id of the image in the original image dataset.
	num = int(num[0])
	label = int((num-1)/Number)
	# print('num: {}, Number: {}, label: {}'.format(num, Number, label))
	# example defines the data format.
	example = tf.train.Example(features=tf.train.Features(feature={  
	"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),  
	'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))  
	}))  
	
	# Serialize the data to string and write it.
	train_aug_writer.write(example.SerializeToString())
	print('Image Name: {}, label: {}'.format(name, label)) 

# write the noised train data.
for name in trainNoise_dir:  
	# the path of an image.
	image_path = cwd + "/TrainNoise/"+name
	# read the image.
	img = Image.open(image_path)  
	# resize the image.
	img = img.resize((ImageWidth, ImageHeight))  
	# convert the image to bytes(raw data).
	img_raw = img.tobytes()
	# calculate the label of this image.
	num = name.split('.')
	# num is the id of the image in the original image dataset.
	num = int(num[0])
	label = int((num-1)/Number)
	# print('num: {}, Number: {}, label: {}'.format(num, Number, label))
	# example defines the data format.
	example = tf.train.Example(features=tf.train.Features(feature={  
	"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),  
	'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))  
	}))  
	
	# Serialize the data to string and write it.
	train_aug_writer.write(example.SerializeToString())
	print('Image Name: {}, label: {}'.format(name, label)) 	
	
# close the TFRecords writer.	
train_aug_writer.close()  
