#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import tensorflow as tf

class JModel():
    def __init__(self,sequence_length,max_words_length,vocab_size,wordembedding_size,char_size,charembedding_size,char_biunits,POS_biunits):
        self.input_word=tf.placeholder(tf.int32,[None,sequence_length],name='input_word')
        self.input_character=tf.placeholder(tf.int32,[None,sequence_length,max_words_length],name='input_character')
        self.input_tag=tf.placeholder(tf.int32,[None,sequence_length],name='POS_tag')
        self.input_arc=tf.placeholder(tf.int32,[None,sequence_length],name='arc')
        self.input_arclabel=tf.placeholder(tf.int32,[None,sequence_length],name='arc label')

        #word embeeding
        with tf.name_scope("word embedding"):
            self.word_embedding=tf.Variable(tf.random_uniform([vocab_size,wordembedding_size],-1.0,1.0))
            self.word_embedded=tf.nn.embedding_lookup(self.word_embedding,self.input_word)  #[batch_size,sequence_length,wordembedding_size]

        #characer-level word embedding
        with tf.name_scope("character-level word embedding"):
            char_input=tf.reshape(self.input_character,[-1,max_words_length]) #
            self.char_embedding=tf.Variable(tf.random_uniform([char_size,charembedding_size],-1.0,1.0))
            char_input=tf.nn.embedding_lookup(self.char_embedding,chat_input) #[batch_size*sequence_length,max_words_length,chatembedding_size]
            with tf.name_scope("char-bilstm"):
                fwlstm=tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(char_biunits),output_keep_prob=self.dropout_keep_prob)
                bwlstm=tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(chat_biunits),output_keep_prob=self.dropout_keep_prob)

                outputs,_=tf.nn.bidirectional_dynamic_rnn(fwlstm,bwlstm,self.input_character)
                charbioutput=tf.concat(outputs,2) #[batch_size*sequence_length,char_biunits*2]
            charoutput=self.MLP(charbioutput,chat_biunits*2,wordembedding_size)
            self.char_embedded=tf.reshape([-1,sequence_length,wordembedding_size])
            
        self.embedded=tf.concat()
        #POS tagging component
        with tf.name_scope("POS tagging component"):

            with tf.name_scope("POS-bilstm"):
                fwlstm=tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(POS_biunits),output_keep_prob=self.dropout_keep_prob)
                bwlstm=tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(POS_biunits),output_keep_prob=self.dropout_keep_prob)

                outputs,_=tf.nn.bidirectional_dynamic_rnn(fwlstm,bwlstm,self.embedded)            


        #Parsing component
        with tf.name_scope("parsing component"):

            with tf.name_scope("arc"):
                pass
            
            with tf.name_scope("arc label"):
                pass
            
            pass
        
    def MLP(self,input,input_size,output_size):
        W = tf.get_variable("W",shape=[input_size, output_size],initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(tf.constant(0.1, shape=[num_classes]), name="b")
        return tf.nn.xw_plus_b(input,W,b)
