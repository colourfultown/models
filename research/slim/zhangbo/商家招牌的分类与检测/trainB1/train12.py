import tensorflow as tf
import os
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3
import time
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
            'label':tf.FixedLenFeature([],tf.int64),
        }
    )
    img=tf.decode_raw(features['image_raw'],tf.uint8)
    # print(img.eavl().shape)
    # img.set_shape(box[0]*box[1]*box[2])
    img = tf.reshape(img, box)
    img=tf.cast(img,tf.float32)*(1.0/255)-0.5

    label=tf.cast(features['label'],tf.int64)
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
    batch_size1=32

    ip = '../createDataB/output/trainA5/'
    ipts = '../createDataB/output/validaA2/'

    batch_size2 = 75
    tsn=int(300 / batch_size2)

    files = tf.train.match_filenames_once(ip + 'data.tfrecords-*')
    filename_qu = tf.train.string_input_producer(files, shuffle=True)
    img, label = read_de(filename_qu)
    imgs, labels = tf.train.shuffle_batch([img, label], batch_size=batch_size1, num_threads=16,
                                          capacity=2000 + 10 * batch_size1, min_after_dequeue=2000)

    tsfiles = tf.train.match_filenames_once(ipts + 'data.tfrecords*')
    tsfilename_qu = tf.train.string_input_producer(tsfiles)
    tsimg, tslabel = read_de(tsfilename_qu)
    tsimgs, tslabels = tf.train.batch([tsimg, tslabel], batch_size=batch_size2, num_threads=10,
                                              capacity=100 + 4 * batch_size2)

    tfimages = tf.placeholder(tf.float32, [None, 299, 299, 3], name='input_images')
    tflabels = tf.placeholder(tf.int64, [None], name='labels')
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, _ = inception_v3.inception_v3(tfimages, num_classes=N_CLASSES, is_training=True)

    tf.losses.softmax_cross_entropy(tf.one_hot(tflabels, N_CLASSES), logits, weights=1.0)
    total_loss = tf.losses.get_total_loss()

    # global_step=tf.Variable(0,trainable=False)
    # LEARNING_RATE=tf.train.exponential_decay(init_LEARNING_RATE,global_step=global_step,
    #                                          decay_steps=500,decay_rate=0.9)
    # add_global=global_step.assign_add(1)

    # train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(total_loss)
    train_step = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(total_loss)
    # 计算正确率。
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tflabels)
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # 定义加载Google训练好的Inception-v3模型的Saver。
    load_fn = slim.assign_from_checkpoint_fn(CKPT_FILE,get_tuned_variables(),ignore_missing_vars=True)
    # 定义保存新模型的Saver。
    saver = tf.train.Saver()

    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # with tf.Session(config=config) as sess:
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        # threads = tf.train.start_queue_runners(sess, coord=coord)
        # sess.run(init)
        # # 加载谷歌已经训练好的模型。
        print('Loading tuned variables from %s' % CKPT_FILE)
        load_fn(sess)

        for i in range(STEPS):
            trX, trY =sess.run([imgs, labels])
            # g,rate=sess.run([add_global,LEARNING_RATE])
            _, loss = sess.run([train_step, total_loss], feed_dict={tfimages:trX,tflabels:trY})
            if i % 30 == 0 or i + 1 == STEPS:
                saver.save(sess, TRAIN_FILE, global_step=i)
                # print(g,rate)
                s=0
                for j in range(tsn):
                    tsX, tsY = sess.run([tsimgs, tslabels])
                    s+= sess.run(evaluation_step, feed_dict={tfimages: tsX, tflabels: tsY})
                validation_accuracy=s/tsn
                print('{} train loss : {},val acc : {}\n'.format(i,loss,validation_accuracy))
        coord.request_stop()
        coord.join(threads)

# 不需要从谷歌训练好的模型中加载的参数。
CHECKPOINT_EXCLUDE_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogits'
# 需要训练的网络层参数明层，在fine-tuning的过程中就是最后的全联接层。
TRAINABLE_SCOPES='InceptionV3/Logits,InceptionV3/AuxLogit'
CKPT_FILE='../model/inception_v3.ckpt'
TRAIN_FILE='model12/A5_1.ckpt'
STEPS=1000000
if __name__=='__main__':
    trian()