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
'''
tf.enable_eager_execution()
array=[[[1,1,1],
        [2,2,2]],
       [[3,4,5],
        [6,7,8]],
       [[9,10,11],
        [12,13,14]]]
x=tf.constant(array,tf.int32)
x=tf.constant([1,2,3,4])

print(x.numpy())
'''
'''
array=[[[1,2,3],
        [4,5,6]],
       [[7,8,9],
        [10,11,12]],
       [[13,14,15],
        [16,17,18]]]
x=tf.constant(array,dtype=tf.int32)
with tf.Session() as sess:
    sp=x.shape
    x1=tf.tile(x,[1,1,sp[1]])
    x2=tf.reshape(x1,[sp[0],sp[1]*2,sp[2]])
    x3=tf.tile(x,[1,sp[1],1])
    x4=tf.concat([x2,x3],-1)
    print(sess.run(x4))
'''
class tmodel:
    def __init__(self):
        self.a=tf.placeholder(tf.int32,[None],name="a")
        self.b=tf.placeholder(tf.int32,[None],name="b")

    def test(self,a,b):
        feed_dict={self.a:a,self.b:b}
        self.judge=tf.less(self.a,self.b)
        if self.judge==True:
            return 'a'
        else:
            return 'b'

x=[1]
y=[2]
with tf.Session() as sess:
    model=tmodel()
    s=model.test(x,y)
    print(s)

