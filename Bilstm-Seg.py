import tensorflow as tf

class SegModel():
    def __init__(self,rnn_size,rnn_layers,embedding_dim,learning_rate,class_num,seq_length,word2id,max_gradients_norm):
        self.rnn_size=rnn_size
        self.rnn_layers=rnn_layers
        self.embedding_dim=embedding_dim
        self.learning_rate=learning_rate
        self.class_num=class_num
        self.word2id=word2id
        self.max_gradients_norm=max_gradients_norm
        self.vocab_size=len(self.word2id)
        self.seq_length=seq_length

        self.input=tf.placeholder(tf.int32,[None,None],name='input')
        self.target=tf.placeholder(tf.int32,[None,None],name='target')
        self.target_length=tf.placeholder(tf.int32,[None],name='target_length')
        self.batch_size=tf.placeholder(tf.int32,[],name='batch_size')
        self.keep_prob=tf.placeholder(tf.float32,name='keep_prob')

        self.build_bilstm()

    def RNNcell(self):
        def single():
            single_cell = tf.contrib.rnn.LSTMCell(self.rnn_size)
            cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=self.keep_prob)
            return cell
        return tf.contrib.rnn.MultiRNNCell([single() for _ in range(self.rnn_layers)])

    def bi_lstmcell(self):
        lstm_fw_cell=tf.contrib.rnn.BasicLSTMCell(self.rnn_size,forget_bias=1.0)
        lstm_bw_cell=tf.contrib.rnn.BasicLSTMCell(self.rnn_size,forget_bias=1.0)
        lstm_fw_cell=tf.contrib.rnn.DropoutWrapper(lstm_fw_cell,output_keep_prob=self.keep_prob)
        lstm_bw_cell=tf.contrib.rnn.DropoutWrapper(lstm_bw_cell,output_keep_prob=self.keep_prob)
        return lstm_fw_cell,lstm_bw_cell

    def build_bilstm(self):
        embedding=tf.Variable(tf.truncated_normal(shape=[self.vocab_size,self.embedding_dim],name='embedding'))
        embedded=tf.nn.embedding_lookup(embedding,self.input)
        cell_fw,cell_bw=self.bi_lstmcell()
        outputs,_=tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,cell_bw=cell_bw,inputs=embedded,dtype=tf.float32)

        output_fw,output_pw=outputs
        output=tf.concat([output_fw,output_pw],axis=-1)
        output = tf.reshape(output, [-1, 2*self.rnn_size])
        Weight = tf.Variable(tf.truncated_normal(shape=[2*self.rnn_size, self.class_num], stddev=0.1))
        Bias = tf.Variable(tf.constant(0.1, shape=[self.class_num]))

        logit = tf.matmul(output, Weight) + Bias  # logit:[batch_size*seq_length,class_num]
        logit = tf.reshape(logit, [self.batch_size, self.seq_length, self.class_num])
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(logit, self.target, self.target_length)
        decode_tags, best_score = tf.contrib.crf.crf_decode(logit, transition_params, self.target_length)
        self.pre=decode_tags
        self.loss = tf.reduce_mean(-log_likelihood)
        self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        self.saver = tf.train.Saver(tf.global_variables())


    def train(self,sess,batch):
        feed_dict={self.input:batch.input,
                   self.target:batch.target,
                   self.target_length:batch.target_length,
                   self.keep_prob:0.5,
                   self.batch_size:len(batch.input)}
        _,loss=sess.run([self.train_op,self.loss],feed_dict=feed_dict)
        return loss

    def demo(self,sess,batch):
        feed_dict={self.input:batch.input,
                   self.target_length:batch.target_length,
                   self.keep_prob:0.5,
                   self.batch_size:1}
        pre=sess.run(self.pre,feed_dict=feed_dict)
        return pre
