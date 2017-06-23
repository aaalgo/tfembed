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
from tensorflow.python.framework import meta_graph
import _tfembed

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', None, '')
flags.DEFINE_string('output', None, '')

def main (_):
    #X = tf.placeholder(tf.float32, shape=(None, stream.dim()), name="X")
    params = ['fc1/weights:0',
              'fc1/biases:0',
              'fc2/weights:0',
              'fc2/biases:0',

              #'fully_connected_1/weights:0',
              #'fully_connected_1/biases:0',
              #'fully_connected_2/weights:0',
              #'fully_connected_2/biases:0',
              #'encode/fully_connected_2/BiasAdd:0'
              'encode/fc2/BiasAdd:0'
             ]

    mg = meta_graph.read_meta_graph_file(FLAGS.model + '.meta')
    X = tf.placeholder(tf.float32, shape=(None, None), name="X")
    params = tf.import_graph_def(mg.graph_def, name='tfembed', input_map={'X:0':X}, return_elements=params)
    sigmoid = params.pop()

    saver = tf.train.Saver(saver_def=mg.saver_def, name='tfembed')
    init = tf.global_variables_initializer()

    #if True:
    #    for n in mg.graph_def.node:
    #        print(n.name)
    config = tf.ConfigProto(device_count = {'GPU': 0})
    with tf.Session(config=config) as sess:
        sess.run(init)
        saver.restore(sess, FLAGS.model)
        params = sess.run(params)
        din, _ = params[0].shape
        #v = np.array([range(din)], dtype=np.float32)
        v = np.array([[0.015625]*din], dtype=np.float32)
        #v /= np.sum(v)
        v = sess.run(sigmoid, feed_dict={X: v})
    #py_v = list(v[0])
    py_v = v[0]
    cpp_v = _tfembed.save_model(FLAGS.output, params)[0]
    print(py_v - cpp_v)
    print(zip(list(py_v), list(cpp_v)))
    pass

if __name__ == '__main__':
    tf.app.run()

