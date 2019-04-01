# -*- coding: utf-8 -*-
# @Time    : 2019/3/28 下午8:11
# @Author  : dabo
# @Email   : dazhangbo_01@163.com
# @File    : run_this2.py
# @Software: PyCharm

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import inception_v4
import os,sys

flags = tf.app.flags
flags.DEFINE_string("input_dir", default="/home/zhangbo/data/cifar-10-batches-py/", help="input-dir")
flags.DEFINE_integer("batch_size", default=32, help="batch-size")
flags.DEFINE_integer("epoch", default=1, help="epoch")

FLAGS = flags.FLAGS



_IMAGE_SIZE = 32
box = [32, 32, 3]
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
labels_to_names = {
    0:'airplane',
    1:'automobile',
    2:'bird',
    3:'cat',
    4:'deer',
    5:'dog',
    6:'frog',
    7:'horse',
    8:'ship',
    9:'truck',
}
N_CLASSES = 10
CHECKPOINT_EXCLUDE_SCOPES = "InceptionV4/Logits,InceptionV4/AuxLogits"

def get_tuned_variables():
    # 返回要加载的参数
    exclusions = [scope.strip() for scope in CHECKPOINT_EXCLUDE_SCOPES.split(",")]
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore

def read_tfrecord_tf():
    input_dir = FLAGS.input_dir
    batch_size = FLAGS.batch_size
    files = tf.train.match_filenames_once(input_dir + 'train_v2.tfrecord') #可以输入一个list
    filename_qu = tf.train.string_input_producer(files, shuffle=True, num_epochs=1) #生成文件名队列
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_qu)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/class/label': tf.FixedLenFeature([], tf.int64),
        }
    )
    img = tf.decode_raw(features['image/encoded'], tf.uint8)
    img = tf.reshape(img, box)
    img = tf.image.resize_images(img, [299, 299], method=0)
    img = tf.cast(img, tf.float32) * (1.0 / 255) - 0.5
    label = tf.cast(features['image/class/label'], tf.int64)
    imgs, labels = tf.train.shuffle_batch([img, label], batch_size=batch_size, num_threads=10,
                                          capacity=10 * batch_size, min_after_dequeue=200)
    return imgs, labels

def read_tfrecord_slim():
    input_dir = FLAGS.input_dir
    batch_size = FLAGS.batch_size
    file_pattern = os.path.join(input_dir, "train.tfrecord")
    # 第一步
    # 将example反序列化成存储之前的格式。由tf完成
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/class/label': tf.FixedLenFeature(
            [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }
    # 第一步
    # 将反序列化的数据组装成更高级的格式。由slim完成
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }
    # 解码器，进行解码
    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)
    # dataset对象定义了数据集的文件位置，解码方式等元信息
    SPLITS_TO_SIZES = {'train': 50000, 'test': 10000}
    _NUM_CLASSES = 10
    _ITEMS_TO_DESCRIPTIONS = {
        'image': 'A [32 x 32 x 3] color image.',
        'label': 'A single integer between 0 and 9',
    }
    dataset = slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=tf.TFRecordReader,
        decoder=decoder,
        num_samples=SPLITS_TO_SIZES['train'],  # 训练数据的总数
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        num_classes=_NUM_CLASSES,
        labels_to_names=labels_to_names  # 字典形式，格式为：id:class_call,
    )
    # provider对象根据dataset信息读取数据
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=1,
        common_queue_capacity=20 * batch_size,
        common_queue_min=10 * batch_size)

    # 获取数据，获取到的数据是单个数据，还需要对数据进行预处理，组合数据
    [image, label] = provider.get(['image', 'label'])
    # 图像预处理
    # image = image_preprocessing_fn(image, _IMAGE_SIZE, _IMAGE_SIZE)

    images, labels = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=10,
        capacity=5 * batch_size)
    labels = slim.one_hot_encoding(
        labels, _NUM_CLASSES - 0)
    batch_queue = slim.prefetch_queue.prefetch_queue(
        [images, labels], capacity=2 * 1)
    # 组好后的数据
    images, labels = batch_queue.dequeue()
    return images, labels

def generate_parse_fn():
    def _sparse_func(example_proto):
        keys_to_features = {}
        keys_to_features["image/encoded"] = tf.FixedLenFeature([], tf.string)
        keys_to_features["image/class/label"] = tf.FixedLenFeature([], tf.int64)
        features = tf.parse_example(example_proto, keys_to_features)
        img = features["image/encoded"]
        img = tf.image.resize_images(img, [299, 299], method=0)
        img = tf.cast(img, tf.float32) * (1.0 / 255) - 0.5
        label = tf.cast(features['image/class/label'], tf.int64)
        return img, label
    return _sparse_func
def input_with_dataset():
    input_dir = FLAGS.input_dir
    batch_size = FLAGS.batch_size
    epoch = FLAGS.epoch
    file_names = [input_dir + 'train_v2.tfrecord']
    files = tf.convert_to_tensor(file_names, dtype=tf.string)
    files = tf.reshape(files, [-1], name="flat_filenames")
    files = tf.data.Dataset.from_tensor_slices(files)
    dataset = files.apply(tf.contrib.data.parallel_interleave(
        tf.data.TFRecordDataset,
        cycle_length=4 * 2))
    dataset = dataset.shuffle(buffer_size=batch_size * 4)  # 打乱时使用的buffer的大小 什么意思
    dataset = dataset.prefetch(buffer_size=batch_size)
    dataset = dataset.repeat(epoch)
    # dataset = dataset.batch(batch_size)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    _parse_fn = generate_parse_fn()
    dataset = dataset.map(_parse_fn, num_parallel_calls=4)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()
def run1():
    CKPT_FILE = "./model/inception_v4.ckpt"
    SAVE_FILE = "./log/inv4.ckpt"

    tfimages = tf.placeholder(tf.float32, [None, 299, 299, 3], name='input_images')
    tflabels = tf.placeholder(tf.int64, [None], name='labels')
    with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
        logits, _ = inception_v4.inception_v4(tfimages, num_classes=N_CLASSES, is_training=True)

    tf.losses.softmax_cross_entropy(tf.one_hot(tflabels, N_CLASSES), logits, weights=1.0)
    total_loss = tf.losses.get_total_loss()
    train_step = tf.train.RMSPropOptimizer(0.0001).minimize(total_loss)
    imgs, labels = read_tfrecord_tf()
    load_fn = slim.assign_from_checkpoint_fn(CKPT_FILE, get_tuned_variables(), ignore_missing_vars=True)
    # 定义保存新模型的Saver。
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()# 开启一个协调器
        threads = tf.train.start_queue_runners(sess=sess, coord=coord) # 使用start_queue_runners 启动队列填充
        # # 加载谷歌已经训练好的模型。
        print('Loading tuned variables from %s' % CKPT_FILE)
        load_fn(sess)
        STEPS = 10000
        try:
            for i in range(STEPS):
                trX, trY = sess.run([imgs, labels])
                _, loss = sess.run([train_step, total_loss], feed_dict={tfimages: trX, tflabels: trY})
                if i % 300 == 0 or i + 1 == STEPS:
                    saver.save(sess, SAVE_FILE, global_step=i)
                print("step-{0}-loss-{1}".format(i, loss))
        except tf.errors.OutOfRangeError:
            print("all-files-end")
        finally:
            coord.request_stop()# 协调器coord发出所有线程终止信号
        coord.join(threads)#把开启的线程加入主线程，等待threads结束


def run2():
    CKPT_FILE = "./model/inception_v4.ckpt"
    SAVE_FILE = "./log/inv4.ckpt"
    tfimages, tflabels = input_with_dataset()
    with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
        logits, _ = inception_v4.inception_v4(tfimages, num_classes=N_CLASSES, is_training=True)

    tf.losses.softmax_cross_entropy(tf.one_hot(tflabels, N_CLASSES), logits, weights=1.0)
    total_loss = tf.losses.get_total_loss()
    train_step = tf.train.RMSPropOptimizer(0.0001).minimize(total_loss)
    load_fn = slim.assign_from_checkpoint_fn(CKPT_FILE, get_tuned_variables(), ignore_missing_vars=True)
    # 定义保存新模型的Saver。
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        load_fn(sess)
        step = 0
        try:
            while True:
                _, loss = sess.run([train_step, total_loss])
                if step % 300 == 0:
                    saver.save(sess, SAVE_FILE, global_step=step)
                    print("step-{0}-loss-{1}".format(step, loss))
                step += 1
        except tf.errors.OutOfRangeError:
            print("out of range at step : {0}".format(step))
        finally:
            print("train-over")


def main(_):
    run2()


if __name__ == "__main__":
    tf.app.run()