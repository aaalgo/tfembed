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

def encode (X, dim):
    # stride is  2 * 2 * 2 * 2 = 16
    net = X
    layers = [X]
    with tf.name_scope('encode'):
        net = slim.fully_connected(net, 512, scope='fc1')
        net = slim.fully_connected(net, dim, activation_fn=tf.sigmoid, scope='fc2')
        net -= 0.5
    net = tf.identity(net, 'hash')
    return net

def triplet_loss (H, margin):
    A = tf.slice(H, [0, 0], [1, -1])    # ref
    B = tf.slice(H, [1, 0], [1, -1])    # near
    C = tf.slice(H, [2, 0], [1, -1])    # far
    l1 = tf.nn.l2_loss(A-B)
    l2 = tf.nn.l2_loss(A-C)
    loss = tf.clip_by_value(l1 - l2 + margin, 0, 1e10)
    return loss, l1, l2

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('db', 'db', '')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
#flags.DEFINE_bool('decay', False, '')
#flags.DEFINE_float('decay_rate', 10000, '')
#flags.DEFINE_float('decay_steps', 0.9, '')
flags.DEFINE_string('model', 'model', 'Directory to put the training data.')
flags.DEFINE_string('resume', None, '')
flags.DEFINE_integer('max_steps', 2000000, '')
flags.DEFINE_integer('epoch_steps', 10000, '')
flags.DEFINE_integer('ckpt_epochs', 20, '')
flags.DEFINE_integer('verbose', logging.INFO, '')
flags.DEFINE_integer('dim', 256, '')
flags.DEFINE_float('margin', 5.0, '')


def main (_):
    logging.basicConfig(level=FLAGS.verbose)
    try:
        os.makedirs(FLAGS.model)
    except:
        pass
    assert FLAGS.db and os.path.exists(FLAGS.db)

    stream = _tfembed.SampleStream(FLAGS.db, FLAGS.db + ".cache")

    X = tf.placeholder(tf.float32, shape=(None, stream.dim()), name="X")

    H = encode(X, dim = FLAGS.dim)

    L, L0, L1 = triplet_loss(H, FLAGS.margin)

    rate = 0.0001
    global_step = tf.Variable(0, name='global_step', trainable=False)
    #if FLAGS.decay:
    #    rate = tf.train.exponential_decay(rate, global_step, FLAGS.decay_steps, FLAGS.decay_rate, staircase=True)
    #    tf.summary.scalar('learning_rate', rate)
    optimizer = tf.train.AdamOptimizer(rate)

    train_op = optimizer.minimize(L, global_step=global_step)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
    config = tf.ConfigProto(device_count = {'GPU': 0})

    metrics = ['loss', 'l0', 'l1', 'accu', 'fill']

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
                h, l, l0, l1, _ = sess.run([H, L, L0, L1, train_op], feed_dict={X: triplet})
                ok, f = _tfembed.eval_mask(h);
                avg += np.array([l, l0, l1, ok, f], dtype=np.float32)
                step += 1
                pass
            avg /= FLAGS.epoch_steps
            txt = ', '.join(['%s=%.4f' % (a, b) for a, b in zip(metrics, list(avg))])
            stop_time = time.time()
            print('step %d %s' % (step, txt))
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

