#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import tensorflow as tf

class JModel():
    def __init__(self,sequence_length,max_words_length,vocab_size,wordembedding_size,char_size,charembedding_size,char_biunits,POS_biunits,num_POS,POSembedding_size):
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
            charoutput=self.MLP(charbioutput,char_biunits*2,wordembedding_size,'char_embedding')
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
            self.tagout=self.MLP(tagoutput,POS_biunits*2,num_POS,'POS-tag')
            self.postag=tf.nn.softmax(self.tagout)#[batch_size,sequence_length,num_POS]

            self.loss1=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.input_tag,logits=self.postag))
            
        self.pos=tf.argmax(self.postag,-1)     #形状应为[batch_size,sequence_length,1] 预测的每个单词的标签
        #Parsing component
        with tf.name_scope("parsing component"):
            POS_embedding=tf.Variable(tf.random_uniform([num_POS,POSembedding_size],-1.0,1.0))
            self.POS_embedded=tf.nn.embedding_lookup(POS_embedding,self.pos)
            self.parinput=tf.concat([self.POS_embedded,self.embedded],-1)
            self.parinput=tf.concat([self.parinput,self.position],-1)
            with tf.name_scope("parse-bilstm"):
                fwlstm=tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(char_biunits),output_keep_prob=self.dropout_keep_prob)
                bwlstm=tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(char_biunits),output_keep_prob=self.dropout_keep_prob)

                outputs,_=tf.nn.bidirectional_dynamic_rnn(fwlstm,bwlstm,self.input_character)

            with tf.name_scope("arc"):
                pass
            
            with tf.name_scope("arc label"):
                pass
            
            pass
        
    def MLP(self,inputs, insize, outsize, scope_name,activation_function=None):
        with tf.variable_scope(scope_name):
            Weights = tf.get_variable("Weights", [insize, outsize], initializer = tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable("Bias", [outsize], initializer = tf.zeros_initializer())
            out = tf.matmul(inputs, Weights) + bias
            if activation_function is None:
                return out
            else:
                return activation_function(out)