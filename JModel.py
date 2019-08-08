#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf
from CLE import MST,GetScore

class JModel():
    def __init__(self,FLAGS):
        self.input_word=tf.placeholder(tf.int32,[None,None],name='input_word')
        self.input_character=tf.placeholder(tf.int32,[None,None,None],name='input_character')
        self.input_tag=tf.placeholder(tf.int32,[None,None,None],name='POS_tag')
        self.input_arc=tf.placeholder(tf.int32,[None,None],name='arc')
        self.input_arclabel=tf.placeholder(tf.int32,[None,None],name='arc_label')
        self.position=tf.placeholder(tf.float32,[None,None,1],name='context_position')
        self.dropout_keep_prob=tf.placeholder(tf.float32,name='dropout_keep_prob')

        #word embeeding
        with tf.name_scope("word_embedding"):
            self.word_embedding=tf.Variable(tf.random_uniform([FLAGS.vocab_size,FLAGS.wordembedding_size],-1.0,1.0))
            self.word_embedded=tf.nn.embedding_lookup(self.word_embedding,self.input_word)  #[batch_size,sequence_length,wordembedding_size]

        #characer-level word embedding
        with tf.name_scope("character-level_word_embedding"):
            char_input=tf.reshape(self.input_character,[-1,FLAGS.max_words_length]) #
            self.char_embedding=tf.Variable(tf.random_uniform([FLAGS.char_size,FLAGS.charembedding_size],-1.0,1.0))
            char_input=tf.nn.embedding_lookup(self.char_embedding,char_input) #[batch_size*sequence_length,max_words_length,charembedding_size]
            with tf.variable_scope("char-bilstm"):
                fwlstm=tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(FLAGS.char_biunits),output_keep_prob=self.dropout_keep_prob)
                bwlstm=tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(FLAGS.char_biunits),output_keep_prob=self.dropout_keep_prob)

                outputs,_=tf.nn.bidirectional_dynamic_rnn(fwlstm,bwlstm,char_input,dtype=tf.float32)
                out_fw,out_bw=outputs
                out_fw=out_fw[:,-1,:]
                out_bw=out_bw[:,-1,:]
                charbioutput=tf.concat([out_fw,out_bw],1) #[batch_size*sequence_length,1,char_biunits*2]
                charbioutput=tf.reshape(charbioutput,[-1,FLAGS.char_biunits*2])
            charoutput=self.MLP(charbioutput,FLAGS.char_biunits*2,FLAGS.wordembedding_size,'char_embedding',tf.nn.leaky_relu)
            self.char_embedded=tf.reshape(charoutput,[-1,FLAGS.sequence_length,FLAGS.wordembedding_size])
        self.embedded=tf.concat([self.char_embedded,self.word_embedded],-1)   #将word-level与char-level的embedding拼接 [batch_size,sequence_length,2*wordembedding_size]

        #POS tagging component
        with tf.name_scope("POS_tagging_component"):
            #self.position=tf.reshape(self.position,[-1,FLAGS.sequence_length,1])
            self.taginput=tf.concat([self.embedded,self.position],2)  #拼接存在问题?
            with tf.name_scope("POS-bilstm"):
                fwlstm=tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(FLAGS.POS_biunits),output_keep_prob=self.dropout_keep_prob)
                bwlstm=tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(FLAGS.POS_biunits),output_keep_prob=self.dropout_keep_prob)

                outputs,_=tf.nn.bidirectional_dynamic_rnn(fwlstm,bwlstm,self.taginput,dtype=tf.float32)
            tagoutput=tf.concat(outputs,2)
            self.tagout=self.MLP(tagoutput,FLAGS.POS_biunits*2,FLAGS.num_POS,'POS-tag',tf.nn.leaky_relu)
            self.postag=tf.nn.softmax(self.tagout)#[batch_size,sequence_length,num_POS]

            self.loss1=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.input_tag,logits=self.postag))
            
        self.pos=tf.argmax(self.postag,-1)     #形状应为[batch_size,sequence_length,1] 预测的每个单词的标签
        self.train1=tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(self.loss1)
        tf.summary.scalar('loss',self.loss1)
        self.summary=tf.summary.merge_all()
        self.saver=tf.train.Saver(tf.global_variables())

        '''
        #Parsing component
        with tf.name_scope("parsing component"):
            POS_embedding=tf.Variable(tf.random_uniform([num_POS,POSembedding_size],-1.0,1.0))
            self.POS_embedded=tf.nn.embedding_lookup(POS_embedding,self.pos)
            self.parinput=tf.concat([self.POS_embedded,self.embedded,self.position],-1)
            with tf.name_scope("parse-bilstm"):
                fwlstm=tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(parse_biunits),output_keep_prob=self.dropout_keep_prob)
                bwlstm=tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(parse_biunits),output_keep_prob=self.dropout_keep_prob)

                outputs,_=tf.nn.bidirectional_dynamic_rnn(fwlstm,bwlstm,self.parinput)
            self.parvec=tf.concat(outputs,2) #[batch_size,sequence_length,parse_biunits*2] vi

            with tf.name_scope("arc"):
                sp=self.parvec.shape
                res=[]
                for i in range(sp[0]):
                    temp=[]
                    for j in range(sp[1]):
                        for k in range(sp[2]):
                            temp.append(tf.concat([self.parvec[i,j],self.parvec[i,k]],tf.float32))
                    res.append(temp)
                self.sc=tf.cast(res,tf.float32) #[batch_size,sequence_length^2,parse_biunits*4] 拼接后的特征
                
                self.score=self.MLP(self.sc,parse_biunits*4,1,'arc',tf.nn.leaky_relu) #[batch_size,sequence_length^2]
                self.score=tf.reshape(self.score,[self.score.shape[0],sequence_length,sequence_length]) #[batch_size,sequence_length,sequence_length]
                score=self.score.eval()
                self.msts,self.maxweights=MST(score)
                self.target_scores=GetScore(score,self.input_arc.eval())
                one=tf.constant(np.ones(self.score.shape[0]),tf.float32)
                zero=tf.constant(0,tf.float32)
                self.loss2=tf.maximum(zero,tf.reduce_mean(tf.add(one,tf.subtract(self.target_scores,self.maxweights))))

            with tf.name_scope("arc label"):
                sp=self.parvec.shape
                res=[]
                for i in range(sp[0]):
                    temp=[]
                    for j in range(sp[1]):
                        head=self.parvec[i,self.msts[i][j]]
                        tail=self.parvec[i,j]
                        temp.append(tf.concat([head,tail],-1))
                    res.append(temp)
                self.arcs=tf.cast(res,tf.float32) #[batch_size,sequence_length,parse_biunits*4]
                self.arclabel=tf.nn.softmax(self.MLP(self.arcs,parse_biunits*4,num_label,'arc_label',tf.nn.leaky_relu))
                self.loss3=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.input_arclabel,logits=self.arclabel))
                self.label=tf.argmax(self.arclabel,-1)

            self.loss=tf.add(tf.add(self.loss1,self.loss2),self.loss3)
            self.train_op=tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
            self.saver = tf.train.Saver(tf.global_variables())
        '''
        
    def MLP(self,inputs, insize, outsize, scope_name,activation_function=None):
        hidsize=128
        with tf.variable_scope(scope_name):
            Weights1 = tf.get_variable("Weights1", [insize, hidsize], initializer = tf.contrib.layers.xavier_initializer())
            bias1 = tf.get_variable("Bias1", [hidsize], initializer = tf.zeros_initializer())
            Weights2 = tf.get_variable("Weights2", [hidsize,outsize], initializer = tf.contrib.layers.xavier_initializer())
            bias2 = tf.get_variable("Bias2", [outsize], initializer = tf.zeros_initializer())
            out1 = tf.matmul(inputs, Weights1) + bias1
            if activation_function is not None:
                out1=activation_function(out1)
            out2 = tf.matmul(out1,Weights2)+bias2
            return out2
        
    def train(self,sess,batch):
        feed_dict={self.input_word:batch.word,
                   self.input_character:batch.char,
                   self.input_tag:batch.tag,
                   self.position:batch.position,
                   self.dropout_keep_prob:0.5}
        _,loss,summary=sess.run([self.train1,self.loss1,self.summary],feed_dict=feed_dict)
        return loss,summary
