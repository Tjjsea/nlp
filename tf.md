#一.随机数
##1.tf.random_normal 
`tf.random_normal(shape,mean=0.0,stddev=1.0,dtype=tf.float32,seed=None,name=None)`
正态分布随机数
mean:均值
stddev:标准差
##2.tf.truncated_normal
`tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None,name=None)`
截断正态分布随机数
只保留[mean-2stddev,mean+2stddev]范围内的随机数
##3.tf.random_uniform
`tf.random_uniform(shape,minval=0,maxval=None,dtype=tf.float32,seed=None,name=None)`
均匀分布随机数
范围为[minval,maxval]
##4.tf.random_shuffle
`tf.random_shuffle(value,seed=None,name=None)`
沿着value的第一维进行随机重新排列

#二.变形
##1.tf.concat
`tf.concat([tensor1, tensor2, tensor3,...], axis)`
##2.tf.reshape
`tf.reshape(tensor,shape,name=None)`
`x=tf.constant([[1,2,3,4],[5,6,7,8]])`
`with tf.Session() as sess:`
    `y=tf.reshape(x,[2,2,2])`
    `print(sess.run(y))`
out:`[[[1 2]`
  `[3 4]]`
 `[[5 6]`
  `[7 8]]]`


#三.损失函数
##1.交叉熵
###1.1 tf.nn.softmax_cross_entropy_with_logits
`tf.nn.softmax_cross_entropy_with_logits(_sentinel=None, labels=None, logits=None, dim=-1, name=None)`
labels和logits有相同的shape，适用于单目标问题，如判断一张图片是猫、狗还是人，即label中只有一个位置对应的是1，其余全为0。output:[batch_size]
###1.2 tf.nn.sigmoid_cross_entropy_with_logits
`tf.nn.sigmoid_cross_entropy_with_logits(_sentinel=None, labels=None, logits=None, name=None)`
labels和logits必须有相同的type和shape，该方法可以用于多目标问题，如判断一张图片中是否包含人、狗、树等，即对应的label包含多个1。但是output不是一个数，而是一个batch中每个样本的loss,所以一般配合tf.reduce_mean(loss)使用。
#四.激活函数
##1.

#五.生成tensor
##1.tf.constant

#六.tensorflow训练

#七.lstm
##1.tf.nn.bidirectional_dynamic_rnn