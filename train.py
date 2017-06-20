#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('build/lib.linux-x86_64-2.7')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import logging
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow.contrib.slim as slim
import _tfembed

def encode (X, dim=64):
    # stride is  2 * 2 * 2 * 2 = 16
    net = X
    layers = [X]
    with tf.name_scope('encode'):
        net = slim.fully_connected(net, 128)
        net = slim.fully_connected(net, dim, activation_fn=None)
        net = tf.sigmoid(net)
    net = tf.identity(net, 'hash')
    return net

def triplet_loss (H, M):
    loss = tf.reduce_mean(tf.abs((H - 0.5) * M))
    loss = tf.identity(loss, 'loss')
    return loss, [loss]

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('db', 'db', '')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_bool('decay', False, '')
flags.DEFINE_float('decay_rate', 10000, '')
flags.DEFINE_float('decay_steps', 0.9, '')
flags.DEFINE_string('model', 'model', 'Directory to put the training data.')
flags.DEFINE_string('resume', None, '')
flags.DEFINE_integer('max_steps', 200000, '')
flags.DEFINE_integer('epoch_steps', 100, '')
flags.DEFINE_integer('ckpt_epochs', 200, '')
flags.DEFINE_integer('verbose', logging.INFO, '')
flags.DEFINE_integer('dim', 64, '')


def main (_):
    logging.basicConfig(level=FLAGS.verbose)
    try:
        os.makedirs(FLAGS.model)
    except:
        pass
    assert FLAGS.db and os.path.exists(FLAGS.db)

    stream = _tfembed.SampleStream(FLAGS.db)

    X = tf.placeholder(tf.float32, shape=(None, stream.dim()), name="X")
    M = tf.placeholder(tf.float32, shape=(None, FLAGS.dim), name="mask")

    H = encode(X, FLAGS.dim)

    loss, metrics = triplet_loss(H, M)

    #tf.summary.scalar("loss", loss)
    metric_names = [x.name[:-2] for x in metrics]
    for x in metrics:
        tf.summary.scalar(x.name.replace(':', '_'), x)

    rate = 0.0001
    global_step = tf.Variable(0, name='global_step', trainable=False)
    #if FLAGS.decay:
    #    rate = tf.train.exponential_decay(rate, global_step, FLAGS.decay_steps, FLAGS.decay_rate, staircase=True)
    #    tf.summary.scalar('learning_rate', rate)
    optimizer = tf.train.AdamOptimizer(rate)

    train_op = optimizer.minimize(loss, global_step=global_step)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:
        sess.run(init)
        if FLAGS.resume:
            saver.restore(sess, FLAGS.resume)
        step = 0
        epoch = 0
        global_start_time = time.time()
        while step < FLAGS.max_steps:
            start_time = time.time()
            avg = np.array([0] * len(metrics), dtype=np.float32)
            for _ in tqdm(range(FLAGS.epoch_steps), leave=False):
                triplet = stream.next()
                feed_dict = {X: triplet}
                h = sess.run(H, feed_dict={X: triplet})
                mask, err, _ = _tfembed.eval_mask(h);
                mm, _ = sess.run([metrics, train_op], feed_dict={X: triplet, M: mask})
                avg += np.array(mm)
                step += 1
                pass
            avg /= FLAGS.epoch_steps
            stop_time = time.time()
            txt = ', '.join(['%s=%.4f' % (a, b) for a, b in zip(metric_names, list(avg))])
            print('step %d: elapsed=%.4f time=%.4f, %s'
                    % (step, (stop_time - global_start_time), (stop_time - start_time), txt))
            epoch += 1
            if epoch and (epoch % FLAGS.ckpt_epochs == 0):
                ckpt_path = '%s/%d' % (FLAGS.model, step)
                start_time = time.time()
                saver.save(sess, ckpt_path)
                stop_time = time.time()
                print('epoch %d step %d, saving to %s in %.4fs.' % (epoch, step, ckpt_path, stop_time - start_time))
        pass
    pass

if __name__ == '__main__':
    tf.app.run()

