import tensorflow as tf
import os
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
    # img.set_shape(box)
    img = tf.reshape(img, box)
    img=tf.cast(img,tf.int32)

    label=tf.cast(features['label'],tf.int32)
    return img,label

def trian():
    ip='../createDataB/output/trainA3_2/'
    # ip='../createDataB/output/validaA1/'
    batch_size=32

    fl = []
    for t in os.listdir(ip):
        fl.append(os.path.join(ip, t))
    filename_qu = tf.train.string_input_producer(fl,shuffle=True)
    img, label = read_de(filename_qu)
    imgs, labels = tf.train.shuffle_batch([img, label], batch_size=batch_size, num_threads=3,
                                          capacity=10000 + 4 * batch_size, min_after_dequeue=10000)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord=coord)
        for _ in range(1000):
            img,label=sess.run([imgs, labels])
            print(_,img.shape,label.shape,label)
        coord.request_stop()
        coord.join(threads)

if __name__=='__main__':
    trian()