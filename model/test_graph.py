# -*- coding: utf-8 -*-)
import tensorflow as tf
import os 
from collections import defaultdict
import numpy as np
import random
import multiprocessing
import time
# a = defaultdict(set)
np.set_printoptions(suppress=True)

# ab = range(0,20)
# def test(index, ll):
# 	print("idnex: ",index)
# 	print("ll: ",ll[index])
# 	time.sleep(2)

# cpu_num = multiprocessing.cpu_count()
# print(cpu_num)
# pool = multiprocessing.Pool(processes = 6)
# results = []
# for i in range(len(ab)):
# 	results.append(pool.apply_async(test, (i,ab)))

# pool.close()
# pool.join()
# a =[]
# a.append((0.56,0.56))
# a.append((0.6,0.9))
# b = sorted(a, key=lambda x:x[0], reverse=True) 
# print (b)

# a['a'].add(('1'))

# #aa = a.get('aa',set())
# print(a['a'])

# print('a' in a)

# b=[1,2,1,1]
# print(np.array(b).shape)
# choice = random.randint(0,1)
# print(choice)
# print(a['b'])
# print(a.get('a','111'))
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

# with tf.Session() as sess:
# 	a = tf.Variable([[1,1,1],[2,2,2]], name="a1", dtype=tf.float32)
# 	l2_1 = tf.nn.l2_normalize(a,1)
# 	l2_2 = tf.nn.l2_normalize(a)

# 	sess.run(tf.global_variables_initializer())
# 	l1, l2 = sess.run([l2_1,l2_2])
# 	print(l1)
# 	print(l2)


class model1:
    def __init__(self, sess):
        with tf.variable_scope("model1"):
            self.a1 = tf.Variable(1, name="a1", dtype=tf.float32)
            self.a2 = tf.Variable(1, name="a2", dtype=tf.float32)
            
            self.word = tf.placeholder(tf.float32)
            self.word_init = self.a2.assign(self.word)

            self.sum1 = tf.add(self.a1,self.a2)
            self.sess = sess

            #trainable_params = tf.global_variables()
            self.trainable_params = tf.trainable_variables()
            # print("model1")
            # print(len(trainable_params))
            # for v in trainable_params:
            #     print(v)
            self.sess.run(tf.global_variables_initializer())


    def get_sum(self):
        return self.sess.run(self.sum1)

    def get_para(self):
        return self.trainable_params

class model2:
    def __init__(self, sess):
        with tf.variable_scope("model2"):
            self.b1 = tf.Variable(2, name="b1", dtype=tf.float32)
            self.b2 = tf.Variable(2, name="b2", dtype=tf.float32)

            self.sum2 = tf.add(self.b1,self.b2)
            self.sess = sess

            #trainable_params = tf.global_variables()
            self.trainable_params = tf.trainable_variables()
            # print("model2")
            # print(len(trainable_params))
            # for v in trainable_params:
            #     print(v)
            self.sess.run(tf.global_variables_initializer())


    def get_sum(self):
        return self.sess.run(self.sum2)

    def get_para(self):
        return self.trainable_params
        

g1 = tf.Graph()
g2 = tf.Graph()

sess1 = tf.Session(graph=g1)
sess2 = tf.Session(graph=g1)


with g1.as_default():
    with sess1.as_default():
        t_m1 = model1(sess1)
        print("model1")
        print(sess1.run(t_m1.trainable_params))
        t_m1.sess.run(t_m1.word_init, feed_dict = {t_m1.word: 3.0})
        print(t_m1.get_sum())
    
    with sess2.as_default():
    	# t_m1 = model1(sess2)
    	# print("model1")
    	print(sess2.run(t_m1.get_sum()))
        # sess2 = tf.Session()
        # t_m2 = model2(sess2)
        # print("model2")
        # print(t_m2.get_para())
        # print(t_m2.get_sum())


# # sess1 = tf.Session(graph=g2)
# # with g2.as_default():
# #     with sess1.as_default(),tf.device('/gpu:1'):
# #         t_m2 = model2(sess1)
# #         print("model2")
# #         print(sess1.run(t_m2.trainable_params))

# # print(t_m1.get_para())
# # print(t_m2.get_para())

#         # sess.run(tf.global_variables_initializer())
#         # out = sess.run(t_m1.sum1)
#         # print ('with graph g1, result: {0}'.format(out))
 
# # with g2.as_default():
# #     with tf.Session(graph=g2) as sess2:
# #         t_m2 = model2()
# #         sess.run(tf.global_variables_initializer())
# #         out = sess.run(t_m2.sum2)
# #         print ('with graph g2, result: {0}'.format(out))

# # out = t_m1.get_sum()
# # print ('with graph g1, result: {0}'.format(out))
# # out = t_m2.get_sum()
# # print ('with graph g1, result: {0}'.format(out))


# # 在计算图g1中定义张量和操作
# g1 = tf.Graph()
# g2 = tf.Graph()

# sess1 = tf.Session(graph=g1)
# sess2 = tf.Session(graph=g1)


# with g1.as_default():
#     with sess1.as_default():
#         t_m1 = model1(sess1)
#     with sess2.as_default():
#         t_m2 = model2(sess2)

#     sess1.run(tf.global_variables_initializer())
#     print("model1")
#     print(sess1.run(t_m1.trainable_params))
#     print("model2")
#     print(sess1.run(t_m2.trainable_params))
#         # sess.run(tf.global_variables_initializer())
#         # out = sess.run(t_m1.sum1)
#         # print ('with graph g1, result: {0}'.format(out))
 
# # with g2.as_default():
# #     with tf.Session(graph=g2) as sess2:
# #         t_m2 = model2()
# #         sess.run(tf.global_variables_initializer())
# #         out = sess.run(t_m2.sum2)
# #         print ('with graph g2, result: {0}'.format(out))

# out = t_m1.get_sum()
# print ('with graph g1, result: {0}'.format(out))
# out = t_m2.get_sum()
# print ('with graph g1, result: {0}'.format(out))
 



