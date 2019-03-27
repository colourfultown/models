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
    ipts='../createDataB/output/validaA1/'

    batch_size=32

    files = tf.train.match_filenames_once(ip + 'data.tfrecords-*')
    filename_qu = tf.train.string_input_producer(files, shuffle=True)
    img, label = read_de(filename_qu)
    imgs, labels = tf.train.shuffle_batch([img, label], batch_size=batch_size, num_threads=2,
                                          capacity=10000 + 4 * batch_size, min_after_dequeue=10000)

    tsfiles = tf.train.match_filenames_once(ipts + 'data.tfrecords*')
    tsfilename_qu = tf.train.string_input_producer(tsfiles)
    tsimg, tslabel = read_de(tsfilename_qu)
    tsimgs, tslabels = tf.train.shuffle_batch([tsimg, tslabel], batch_size=batch_size, num_threads=1,
                                          capacity=1000 + 4 * batch_size, min_after_dequeue=100)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord=coord)
        for _ in range(1000):
            a,b,c,d=sess.run([imgs, labels,tsimgs, tslabels])
            print(_,type(a),type(b),a.shape,b.shape,b)
            print(_,type(c),type(d),c.shape,d.shape,d)
        coord.request_stop()
        coord.join(threads)

if __name__=='__main__':
    trian()