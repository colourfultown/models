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

def input(ip,batch_size,n_epochs=10000):
    fl=[]
    for t in os.listdir(ip):
        fl.append(os.path.join(ip,t))
    filename_qu=tf.train.string_input_producer(fl,num_epochs=n_epochs)
    # filename_qu=tf.train.string_input_producer(fl)
    img,label=read_de(filename_qu)
    imgs,labels=tf.train.shuffle_batch([img,label],batch_size=batch_size,num_threads=3,
                                       capacity=1000+4*batch_size,min_after_dequeue=100)
    print('@')
    return imgs,labels

def trian():
    ip='../createDataB/output/trainA3/'
    batch_size=32
    img,label=input(ip,batch_size)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord=coord)
        for _ in range(10):
            img,label=sess.run([img,label])
            print(img.shape,label.shape,label)
        coord.request_stop()
        coord.join(threads)

if __name__=='__main__':
    trian()