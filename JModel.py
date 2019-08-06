#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf
from CLE import MST,GetScore

class JModel():
    def __init__(self,sequence_length,max_words_length,vocab_size,wordembedding_size,char_size,charembedding_size,char_biunits,POS_biunits,num_POS,POSembedding_size,parse_biunits,num_label,learning_rate):
        self.input_word=tf.placeholder(tf.int32,[None,sequence_length],name='input_word')
        self.input_character=tf.placeholder(tf.int32,[None,sequence_length,max_words_length],name='input_character')
        self.input_tag=tf.placeholder(tf.int32,[None,sequence_length,num_POS],name='POS_tag')
        self.input_arc=tf.placeholder(tf.int32,[None,sequence_length],name='arc')
        self.input_arclabel=tf.placeholder(tf.int32,[None,sequence_length],name='arc label')
        self.position=tf.placeholder(tf.float32,[None,sequence_length],name='context position')

        #word embeeding
        with tf.name_scope("word embedding"):
            self.word_embedding=tf.Variable(tf.random_uniform([vocab_size,wordembedding_size],-1.0,1.0))
            self.word_embedded=tf.nn.embedding_lookup(self.word_embedding,self.input_word)  #[batch_size,sequence_length,wordembedding_size]

        #characer-level word embedding
        with tf.name_scope("character-level word embedding"):
            char_input=tf.reshape(self.input_character,[-1,max_words_length]) #
            self.char_embedding=tf.Variable(tf.random_uniform([char_size,charembedding_size],-1.0,1.0))
            char_input=tf.nn.embedding_lookup(self.char_embedding,char_input) #[batch_size*sequence_length,max_words_length,charembedding_size]
            with tf.variable_scope("char-bilstm"):
                fwlstm=tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(char_biunits),output_keep_prob=self.dropout_keep_prob)
                bwlstm=tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(char_biunits),output_keep_prob=self.dropout_keep_prob)

                outputs,_=tf.nn.bidirectional_dynamic_rnn(fwlstm,bwlstm,self.input_character)
                charbioutput=tf.concat(outputs,2) #[batch_size*sequence_length,char_biunits*2]
            charoutput=self.MLP(charbioutput,char_biunits*2,wordembedding_size,'char_embedding',tf.nn.leaky_relu)
            self.char_embedded=tf.reshape([-1,sequence_length,wordembedding_size])
            
        self.embedded=tf.concat([self.word_embedded,self.char_embedded],2)   #将word-level与char-level的embedding拼接 [batch_size,sequence_length,2*wordembedding_size]

        #POS tagging component
        with tf.name_scope("POS tagging component"):
            self.taginput=tf.concat([self.embedded,self.position],-1)  #拼接存在问题?
            with tf.name_scope("POS-bilstm"):
                fwlstm=tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(POS_biunits),output_keep_prob=self.dropout_keep_prob)
                bwlstm=tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(POS_biunits),output_keep_prob=self.dropout_keep_prob)

                outputs,_=tf.nn.bidirectional_dynamic_rnn(fwlstm,bwlstm,self.taginput)
            tagoutput=tf.concat(outputs,2)
            self.tagout=self.MLP(tagoutput,POS_biunits*2,num_POS,'POS-tag',tf.nn.leaky_relu)
            self.postag=tf.nn.softmax(self.tagout)#[batch_size,sequence_length,num_POS]

            self.loss1=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.input_tag,logits=self.postag))
            
        self.pos=tf.argmax(self.postag,-1)     #形状应为[batch_size,sequence_length,1] 预测的每个单词的标签

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

    def get_treescore(self,tree):
        '''
        获取句法树的评分
        '''
        pass