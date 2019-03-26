# -*- coding: utf-8 -*-
# @Time    : 19-3-26 下午10:31
# @Author  : dabo
# @Site    : 
# @File    : run_this.py
# @Software: PyCharm

import tensorflow as tf
import numpy as np
import os,sys
from six.moves import cPickle
tf.app.flags.DEFINE_string("input_dir", default="/home/zhangbo/data/cifar-10-batches-py/", help="input-dir")


FLAGS = tf.app.flags.FLAGS
_IMAGE_SIZE = 32
_CLASS_NAMES = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
]

def int64_feature(values):
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def bytes_feature(values):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def float_feature(values):
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def image_to_tfexample(image_data, image_format, height, width, class_id):
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': bytes_feature(image_data),
      'image/format': bytes_feature(image_format),
      'image/class/label': int64_feature(class_id),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
  }))
def write_into_tfrecord(file_path_in, tf_writer):
    with tf.gfile.Open(file_path_in, 'rb') as f:
        if sys.version_info < (3,):
            data = cPickle.load(f)  # python2
        else:
            data = cPickle.load(f, encoding='bytes')  # python3
    images = data[b'data']
    num_images = images.shape[0]
    images = images.reshape((num_images, 3, 32, 32))
    labels = data[b'labels']
    with tf.Graph().as_default():
        image_placeholder = tf.placeholder(dtype=tf.uint8)
        encoded_image = tf.image.encode_png(image_placeholder)
        with tf.Session('') as sess:
            for j in range(num_images):
                image = np.squeeze(images[j]).transpose((1, 2, 0))
                label = labels[j]
                png_string = sess.run(encoded_image,
                                      feed_dict={image_placeholder: image})
                example = image_to_tfexample(
                    png_string, b'png', _IMAGE_SIZE, _IMAGE_SIZE, label)
                tf_writer.write(example.SerializeToString())
def cifar_to_tfrecord():
    input_dir = FLAGS.input_dir
    train_file_names = ["data_batch_"+str(i) for i in range(1, 6, 1)]
    test_file_names = ["test_batch"]
    file_path_train_out = os.path.join(input_dir, "train.tfrecord")
    tf_writer = tf.python_io.TFRecordWriter(file_path_train_out)
    for index, name in enumerate(train_file_names):
        file_path_in = os.path.join(input_dir, name)
        write_into_tfrecord(file_path_in, tf_writer)
    tf_writer.close()
    file_path_test_out = os.path.join(input_dir, "test.tfrecord")
    tf_writer_test = tf.python_io.TFRecordWriter(file_path_test_out)
    for index, name in enumerate(test_file_names):
        file_path_in = os.path.join(input_dir, name)
        write_into_tfrecord(file_path_in, tf_writer_test)
    tf_writer_test.close()



if __name__ == "__main__":
    cifar_to_tfrecord()