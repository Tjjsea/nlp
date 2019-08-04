import tensorflow as tf

class Seq2SeqModel():
    def __init__(self,rnn_size,rnn_layers,embedding_dim,learning_rate,word2id,mode,max_gradients_norm=0.5):
        self.encoder_input=tf.placeholder(tf.int32,[None,None],name='encoder_input')
        self.encoder_input_length=tf.placeholder(tf.int32,[None],name='encoder_input_length')
        self.decoder_target=tf.placeholder(tf.int32,[None,None],name='decoder_target')
        self.decoder_target_length=tf.placeholder(tf.int32,[None],name='decoder_target_length')
        self.embedding_dim=embedding_dim
        self.vocab_size=len(word2id)
        self.rnn_size=rnn_size
        self.rnn_layers=rnn_layers
        self.keep_prob=tf.placeholder(tf.float32,name='keep_prob')
        self.mode=mode
        self.batch_size=tf.placeholder(tf.int32,[],name='batch_size')
        self.word2id=word2id
        self.learning_rate=learning_rate
        self.max_gradients_norm=max_gradients_norm
        self.max_decoder_target_length=tf.reduce_max(self.decoder_target_length,name='max_decoder_target_length')

        self.buildmodel()

    def RNNcell(self):
        def single():
            single_cell=tf.contrib.rnn.LSTMCell(self.rnn_size)
            cell=tf.contrib.rnn.DropoutWrapper(single_cell,output_keep_prob=self.keep_prob)
            return cell
        return tf.contrib.rnn.MultiRNNCell([single() for _ in range(self.rnn_layers)])

    def buildmodel(self):
        with tf.variable_scope('encoder'):
            encoder_embedding=tf.Variable(tf.truncated_normal(shape=[self.vocab_size,self.embedding_dim],name='encoder_embedding'))
            encoder_embedded=tf.nn.embedding_lookup(encoder_embedding,self.encoder_input)
            encoder_cell=self.RNNcell()
            encoder_outputs,encoder_state=tf.nn.dynamic_rnn(encoder_cell,encoder_embedded,dtype=tf.float32)

        with tf.variable_scope('decoder'):
            '''
            实现decoder,使用tf提供的Basic_Decoder类
            tf.contrib.seq2seq.BasicDecoder.__init__(cell,helper,initial_state,output_layer)
            cell:An RnnCell instance
            helper:A Helper instance
            initial_state:encoder_state
            Helper:decode阶段，决定下一时刻的输入
            训练：tf.contrib.seq2seq.TrainingHelper.__init__(inputs,sequence_length,time_major=False,name=None)
            inputs:decoder的输入，这里应该是target
            sequence_length:当前batch中每个序列的长度
            预测：tf.contrib.seq2seq.GreedyEmbeddingHelper.__init__(embedding,start_tokens,end_token)
            embedding: A callable that takes a vector tensor of ids (argmax ids), or the params argument for embedding_lookup. 
                       The returned tensor will be passed to the decoder input.
            start_tokens: int32 vector shaped [batch_size], the start tokens.
            end_token: int32 scalar, the token that marks end of decoding.
            '''
            decoder_input=tf.strided_slice(self.decoder_target, [0, 0], [self.batch_size, -1], [1, 1])
            self.decoder_input = tf.concat([tf.fill([self.batch_size, 1], self.word2id['<go>']), decoder_input], 1)
            decoder_embedding=tf.Variable(tf.truncated_normal(shape=[self.vocab_size,self.embedding_dim],name='decoder_embedding'))
            decoder_embedded=tf.nn.embedding_lookup(decoder_embedding,self.decoder_input)

            attention_mechanism=tf.contrib.seq2seq.BahdanauAttention(num_units=self.rnn_size,memory=encoder_outputs,memory_sequence_length=self.encoder_input_length)

            decoder_cell=self.RNNcell()
            decoder_cell=tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell,attention_mechanism=attention_mechanism,attention_layer_size=self.rnn_size,
                                                     name='Attention_Wrapper')
            helper=None
            if self.mode=='train':
                helper=tf.contrib.seq2seq.TrainingHelper(decoder_embedded,self.decoder_target_length,name='traininghelper')
            elif self.mode=='inference':
                start_tokens=tf.ones([self.batch_size,],tf.int32)*self.word2id['<go>']
                end_token=self.word2id['<eos>']
                helper=tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embedding,start_tokens=start_tokens,end_token=end_token)

            decoder_initial_state=decoder_cell.zero_state(batch_size=self.batch_size,dtype=tf.float32).clone(cell_state=encoder_state)
            '''
            全连接层
            tf.layers.Dense: f=activation(x*W+b)
            units:输出维度
            activation:激活函数
            use_bias:whether the layer use a bias
            kernel_initializer:权重矩阵W初始化函数
            bias_initializer:bias初始化函数
            '''
            output_layer=tf.layers.Dense(self.vocab_size,kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.1))

            decoder=tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,helper=helper,initial_state=decoder_initial_state,output_layer=output_layer)
            '''
            tf.contrib.dynamic_decode(decoder,output_time_major=False,impute_finished=False,maximum_iterations=None,parallel_iterations=32
                                      swap_memory=False,scope=None)
            decoder: a decoder instance
            output_time_major:决定输出的形状
            return:(final_outputs,final_state,final_sequence_lengths)
                    final_outputs=(rnn_outputs,sample_id)
            '''
            decoder_outputs,_,_=tf.contrib.seq2seq.dynamic_decode(decoder,impute_finished=True,maximum_iterations=None)

            if self.mode=='train':
                self.decoder_logits_train = tf.identity(decoder_outputs.rnn_output)
                #self.decoder_predict_train = tf.expand_dims(tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_pred_train'),-1)
                #print(self.decoder_predict_train)

                mask=tf.sequence_mask(self.decoder_target_length,self.max_decoder_target_length,dtype=tf.float32,name='masks')

                self.loss=tf.contrib.seq2seq.sequence_loss(logits=self.decoder_logits_train,targets=self.decoder_target,weights=mask)

                tf.summary.scalar('loss',self.loss)
                self.summary=tf.summary.merge_all()

                optimizer=tf.train.AdamOptimizer(self.learning_rate)
                trainable_params=tf.trainable_variables()
                gradients=tf.gradients(self.loss,trainable_params)
                clip_gradients, _ =tf.clip_by_global_norm(gradients,self.max_gradients_norm)
                self.train_op = optimizer.apply_gradients(zip(clip_gradients,trainable_params))

            elif self.mode=='inference':
                #self.decoder_predict=tf.expand_dims(decoder_outputs.sample_id,-1)
                #self.decoder_predict = tf.expand_dims(tf.argmax(tf.identity(decoder_outputs.rnn_output), axis=-1, name='decoder_pred'), -1)
                output=tf.reshape(decoder_outputs.rnn_output,[-1,self.vocab_size])
                self.decoder_predict=tf.argmax(output,1)
        self.saver=tf.train.Saver(tf.global_variables())

    def train(self,sess,batch):
        feed_dict={self.encoder_input:batch.encoder_input,
                   self.encoder_input_length:batch.encoder_input_length,
                   self.decoder_target:batch.decoder_target,
                   self.decoder_target_length:batch.decoder_target_length,
                   self.keep_prob:0.5,
                   self.batch_size:len(batch.encoder_input)}
        _,loss,summary=sess.run([self.train_op,self.loss,self.summary],feed_dict=feed_dict)
        return loss,summary

    def infer(self,sess,batch):
        feed_dict={self.encoder_input:batch.encoder_input,
                   self.encoder_input_length:batch.encoder_input_length,
                   self.keep_prob:0.5,
                   self.batch_size:len(batch.encoder_input)}
        predict=sess.run(self.decoder_predict,feed_dict=feed_dict)
        return predict

