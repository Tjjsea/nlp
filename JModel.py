#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import tensorflow as tf

class JModel():
    def __init__(self,sequence_length):
        self.input_x=tf.placeholder(tf.int32,[None,sequence_length],name='input_x')
        self.input_tag=tf.placeholder(tf.int32,[None,sequence_length],name='POS_tag')
        self.input_arc=tf.placeholder(tf.int32,[None,sequence_length],name='arc')
        self.input_arclabel=tf.placeholder(tf.int32,[None,sequence_length],name='arc label')


        #word embeeding
        with tf.name_scope("embedding"):
            self.word_embedding=tf.Variable(tf.random_uniform([vocab_size,embedding_size],-1.0,1.0))