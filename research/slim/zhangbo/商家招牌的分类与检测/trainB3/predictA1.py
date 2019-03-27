import tensorflow as tf
import os
import numpy as np
import tensorflow.contrib.slim as slim
from nets import inception
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

box=[299, 299,3]

def read_de(filename_qu):
    reader=tf.TFRecordReader()
    _,serialized_example=reader.read(filename_qu)
    features=tf.parse_single_example(
        serialized_example,
        features={
            'image_raw':tf.FixedLenFeature([],tf.string),
            'name':tf.FixedLenFeature([],tf.int64),
        }
    )
    img=tf.decode_raw(features['image_raw'],tf.uint8)
    # print(img.eavl().shape)
    # img.set_shape(box)
    img = tf.reshape(img, box)
    img = tf.cast(img, tf.float32) * (1.0 / 255) - 0.5

    label=tf.cast(features['name'],tf.int64)
    return img,label

def get_tuned_variables():
    exclusions = [scope.strip() for scope in CHECKPOINT_EXCLUDE_SCOPES.split(',')]

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


def get_trainable_variables():
    scopes = [scope.strip() for scope in TRAINABLE_SCOPES.split(',')]
    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train

def trian():
    N_CLASSES=100
    LEARNING_RATE=0.0001
    batch_size=100

    ip = '../createDataB/output/testA1/'

    files = tf.train.match_filenames_once(ip + 'data.tfrecords*')
    filename_qu = tf.train.string_input_producer(files)
    img, label = read_de(filename_qu)
    imgs, labels = tf.train.batch([img, label], batch_size=batch_size, num_threads=4,
                                          capacity=500)


    tfimages = tf.placeholder(tf.float32, [None, 299, 299, 3], name='input_images')
    tflabels = tf.placeholder(tf.int64, [None], name='labels')
    with slim.arg_scope(inception.inception_v4_arg_scope()):
        logits, _ = inception.inception_v4(tfimages, num_classes=N_CLASSES, is_training=True)

    trainable_variables = get_trainable_variables()
    tf.losses.softmax_cross_entropy(tf.one_hot(tflabels, N_CLASSES), logits, weights=1.0)
    total_loss = tf.losses.get_total_loss()
    # train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(total_loss)
    train_step = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(total_loss)
    # 计算正确率。
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tflabels)
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 定义保存新模型的Saver。
    saver = tf.train.Saver()
    f=open('result/resA1.txt','w')
    f.write('code,pre\n')

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        saver.restore(sess,TRAIN_FILE)
        print('Loading tuned variables from %s' % TRAIN_FILE)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord=coord)
        for i in range(STEPS):
            print(i)
            X,Y =sess.run([imgs, labels])
            pre = sess.run(logits, feed_dict={tfimages: X})
            r=pre.argmax(axis=1)
            for a,b in zip(Y,r):
                f.write('%d,%d\n'%(a,b+1))


        coord.request_stop()
        coord.join(threads)
    f.close()

# 不需要从谷歌训练好的模型中加载的参数。
CHECKPOINT_EXCLUDE_SCOPES = 'InceptionV4/Logits,InceptionV4/AuxLogits'
# 需要训练的网络层参数明层，在fine-tuning的过程中就是最后的全联接层。
TRAINABLE_SCOPES='InceptionV4/Logits,InceptionV4/AuxLogit'
# CKPT_FILE='../model/inception_v4.ckpt'
TRAIN_FILE='model1_1/max3/A5_1.ckpt-1050'
STEPS=10
if __name__=='__main__':
    trian()