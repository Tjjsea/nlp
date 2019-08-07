#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import tensorflow as tf
from tqdm import tqdm
from JModel import JModel
from GetBatch import getbatch
import os
import math
import json

batch_size=64
sequence_length=36
max_words_length=16
char_size=54
num_POS=17
num_label=38
with open('data/words.json') as fin:
    worddict=json.load(fin)
    vocab_size=len(worddict)

tf.app.flags.DEFINE_integer("sequence_length",sequence_length,"sequence length")
tf.app.flags.DEFINE_integer("max_words_length",max_words_length,"max word's length")
tf.app.flags.DEFINE_integer("vocab_size",vocab_size,"vocab size")
tf.app.flags.DEFINE_integer("char_size",char_size,"char size")
tf.app.flags.DEFINE_integer("wordembedding_size",100,"word level embedding size")
tf.app.flags.DEFINE_integer("charembedding_size",100,"char level embedding size")
tf.app.flags.DEFINE_integer("char_biunits",256,"num of hidden layer for char embedding bilstm")
tf.app.flags.DEFINE_integer("POS_biunits",256,"num of hidden layer in bilstm of tagging component")
tf.app.flags.DEFINE_integer("num_POS",num_POS,"num of POStags")
tf.app.flags.DEFINE_integer("POSembedding_size",100,"POS tag embedding size")
tf.app.flags.DEFINE_integer("parse_biunits",256,"num of hidden layer in bilstm of parsing component")
tf.app.flags.DEFINE_integer("num_label",num_label,"num of arc labels")
tf.app.flags.DEFINE_integer("epochs",1,"num of epochs")
tf.app.flags.DEFINE_integer("batch_size",batch_size,"batch size")
tf.app.flags.DEFINE_integer("steps_per_checkpoint",100,"save model checkpoint every this iteration")
tf.app.flags.DEFINE_float("learning_rate",0.01,"learing rate")
tf.app.flags.DEFINE_string("model_dir","model/POS/","path to save model checkpoints")
tf.app.flags.DEFINE_string("model_name","POStag.ckpt","file name used for model checkpoints")
FLAGS=tf.app.flags.FLAGS

with tf.Session() as sess:
    model=JModel(FLAGS)
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Reloading model parameters..')
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('Created new model parameters..')
        sess.run(tf.global_variables_initializer())
    
    current_step = 0
    summary_writer = tf.summary.FileWriter(FLAGS.model_dir, graph=sess.graph)
    for e in range(FLAGS.epochs):
        print("----- Epoch {}/{} -----".format(e + 1, FLAGS.epochs))

        batches=getbatch('train',batch_size)
        # Tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)。
        for nextBatch in tqdm(batches, desc="Training"):
            loss, summary = model.train(sess, nextBatch)
            current_step += 1
            # 每多少步进行一次保存
            if current_step % FLAGS.steps_per_checkpoint == 0:
                tqdm.write("----- Step %d -- Loss %.2f " % (current_step, loss))
                summary_writer.add_summary(summary, current_step)
                checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
                model.saver.save(sess, checkpoint_path, global_step=current_step)