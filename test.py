import tensorflow as tf

'''
#tensor 迭代，切片
x=tf.constant([[[1,2],[3,4]],[[5,6],[7,8]]])

with tf.Session() as sess:
    sp=x.shape
    res=[]
    for i in range(sp[0]):
        temp=[]
        for j in range(sp[1]):
            for k in range(sp[2]):
                temp.append(tf.concat([x[i,j],x[i][k]],-1))
        res.append(temp)
    res=tf.cast(res,tf.int32)
    print(res.shape)
    print(sess.run(res))
'''

x=tf.constant([1,2],tf.float32)
y=tf.constant([1,2],tf.float32)
z=tf.constant([1,2],tf.float32)
with tf.Session() as sess:
    a=tf.add(x,y)
    print(sess.run(a))