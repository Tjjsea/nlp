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
###1.2 

#四.激活函数
##1.