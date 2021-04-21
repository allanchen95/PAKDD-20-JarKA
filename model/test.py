import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import embed_sequence
from tensorflow.python.util import nest
import os
import multiprocessing as mp
import time
import gc
# a = [1,2,3]
	
# b = np.zeros((3, 4))
# b[np.arange(3), a] = 1
# c = np.array([[0,1,0,0],[1,0,0,0],[0,1,0,0]])
# print(b*c)
# a = np.random.randint(0, 10, size = [10000,20],dtype = np.int8)
# b = np.random.randint(0, 10, size = [10000,20],dtype = np.int8)
# pad_a = np.zeros([10000, 10000, 20, 20], dtype = np.int8)


# cpu_num = multiprocessing.cpu_count()
# pool = multiprocessing.Pool(processes = cpu_num)

# # div via cpu_num
# index = range(10000)
# split_num = int(10000 / cpu_num)
# n = 10000 % cpu_num

# div_key_list = []
# for i in range(cpu_num):
# 	div_key_list.append(index[i*split_num:(i+1)*split_num])
# div_key_list.append(index[cpu_num * split_num:])
# print(len(div_key_list))	




	# def count_num(i, key_list, sub_a, b,pad_a):
	# 	print("i",i)
	# 	for i in range(len(key_list)):
	# 		for j in range(20):
	# 			if(sub_a[i][j]!=0):
	# 				x,y = np.where(b == sub_a[i][j])
	# 				for zz in zip(x,y):
	# 					pad_a[i][zz[0]][j][zz[1]] = 1
	# 	return 1
# pad_a = np.zeros([10000, 50, 1000], dtype = np.int8)
# def count_num(index, key_list,sub_a,bb,pad_aa):
# 	count = 0
# 	print(index)
# 	for i in range(len(key_list)):
# 		for j in range(5):
# 			if(sub_a[i][j]!=0):
# 				x,y = np.where(b == sub_a[i][j])
# 				for zz in zip(x,y):
# 					count+=1
# 					pad_aa[i][zz[0]][j][zz[1]] = 1
# 	return pad_aa, count

# a = np.random.randint(0, 10, size = [10000,20],dtype = np.int8)
# b = np.random.randint(0, 10, size = [10000,20],dtype = np.int8)
# pool = mp.Pool(processes = 8) 

# index = range(10000)
# split_num = int(10000 / 8)
# n = 10000 % 8

# div_key_list = []
# for i in range(8):
# 	div_key_list.append(index[i*split_num:(i+1)*split_num])
# div_key_list.append(index[8 * split_num:])
# print(len(div_key_list))

# s_t = time.time()
# multi_res = []
# for i in range(len(div_key_list)):
# 	key_list = div_key_list[i]
# 	multi_res.append(pool.apply_async(count_num, (i, key_list, a[key_list,:],b,pad_a)))

# pool.close()
# pool.join()
# for res in multi_res:
# 	tmp_a, count = res.get()
# 	print("1", count)
# 	pad_a += tmp_a
# 	del tmp_a
# 	gc.collect()
# print(time.time()-s_t)
# print(len(np.where(pad_a!= 0)[0]))
# def count_num(i, key_list, sub_a, b , pad_a):
# 	for i in range(len(key_list)):
# 		for j in range(20):
# 			if(sub_a[i][j]!=0):
# 				x,y = np.where(b == sub_a[i][j])
# 				for zz in zip(x,y):
# 					pad_a[i][zz[0]][j][zz[1]] = 1


# a = np.random.randint(0, 10, size = [10,5],dtype = np.int8)
# b = np.random.randint(0, 10, size = [10,6],dtype = np.int8)
# #cpu_num = multiprocessing.cpu_count()
# cpu_num = 3
# pool = multiprocessing.Pool(processes = cpu_num)
# pad_a = np.zeros([10, 10, 5, 6], dtype = np.int8)
# # div via cpu_num
# index = range(10)
# split_num = int(10 / cpu_num)
# n = 10 % cpu_num

# div_key_list = []
# for i in range(cpu_num):
# 	div_key_list.append(index[i*split_num:(i+1)*split_num])
# div_key_list.append(index[cpu_num * split_num:])
# print(len(div_key_list))
# s_t = time.time()
# res = []

# for i in range(len(div_key_list)):
# 	key_list = div_key_list[i]
# 	pool.apply_async(count_num,(i, key_list, a[key_list,:],b,pad_a))

# print(len(np.where(pad_a!=0)[0]))
# print(time.time() - s_t)
# for i in range(6):
# 	for j in range(6):
# 		if(a[i][j]!=0):
# 			x,y = np.where(b == a[i][j])
# 			for zz in zip(x,y):
# 				pad_a[i][zz[0]][j][zz[1]] = 1
# print(pad_a)




with tf.device('/gpu:1'):
	with tf.device('/gpu:1'):

		data_input = tf.Variable(tf.random_normal([400,20,100], dtype=tf.float16), name="input")
		embedding = tf.Variable(tf.random_normal([400,20,100], dtype=tf.float16), name="output")
		# dd = tf.transpose(embedding, perm=[1,0])
		res = tf.einsum('aij,xyj->axiy', data_input, embedding)
		# mat = tf.matmul(data_input,dd)
		length = tf.Variable([1,0,0], name="output", dtype=tf.int32)


# length_mask = tf.sequence_mask(length, 3, dtype=tf.int32)
# l_s = tf.shape(length_mask)
# data_embed = tf.nn.embedding_lookup(embedding, data_input)
# # length_maks_re = tf.reshape(length_mask, [l_s[0],ls[1],1])
# length_mask_re = tf.expand_dims(length_mask,-1)

# mask_embed = data_embed * length_mask_re
tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

tf_config.gpu_options.allow_growth = True
with tf.Session(config=tf_config) as sess:
	sess.run(tf.global_variables_initializer())
	a = sess.run(res)


# 	#print(a)
# # 	b = sess.run(data_embed)
# # 	c = sess.run(mask_embed)

	print(np.array(a).shape)
	time.sleep(10)
	del a
	gc.collect()
# 	print(a)
# 	print(np.array(b).shape)
# 	print(b)
# 	print(np.array(c).shape)
# 	print(c)


# def bulid_rnn_cell(rnn_type, rnn_size, rnn_layer, dropout_keep_prob=1):
#     def create_single_rnn():
#         if rnn_type == 'lstm':
#             cell = tf.contrib.rnn.LSTMCell(rnn_size)
#         else:
#             cell = tf.contrib.rnn.GRUCell(rnn_size)

#         cell = tf.contrib.rnn.DropoutWrapper(
#             cell, output_keep_prob=dropout_keep_prob)
#         return cell
#     return tf.contrib.rnn.MultiRNNCell([create_single_rnn() for _ in range(rnn_layer)])

# #data_input = tf.random_normal(shape=[batch_size,3,6], dtype=tf.float32)
# data_input = tf.Variable([[0,1,2],[2,3,4],[1,1,1]], name="input", dtype=tf.int32)
# dec_input = tf.Variable([[0,1,2],[2,3,4],[1,1,1]], name="output", dtype=tf.int32)

# max_len = tf.constant(3)
# embedding = tf.Variable(tf.random_normal([6,6]), name="embedding", dtype=tf.float32)
# emb_data_input = tf.nn.embedding_lookup(embedding, data_input)
# # cell = tf.nn.rnn_cell.BasicLSTMCell(10, forget_bias=1.0, state_is_tuple=True)
# # init_state = cell.zero_state(batch_size, dtype=tf.float32)
# # enc_output, enc_state = tf.nn.dynamic_rnn(cell, data_input, initial_state=init_state) #time_major如果是True，就表示RNN的steps用第一个维度表示，建议用这个，运行速度快一点。

# enc_cell_fw = bulid_rnn_cell('lstm', 10, 4, 0.5)
# enc_cell_bw = bulid_rnn_cell('lstm', 10, 4, 0.5)

# enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(enc_cell_fw, enc_cell_bw, emb_data_input, dtype=tf.float32)

# new_enc_output = tf.concat([enc_output[0], enc_output[1]], -1)

# new_enc_state = []
# for i in range(len(enc_state[0])):
#     new_enc_state_c = tf.concat(
#         [enc_state[0][i].c, enc_state[1][i].c], -1)
#     new_enc_state_h = tf.concat(
#         [enc_state[0][i].h, enc_state[1][i].h], -1)
#     new_enc_state.append(tf.contrib.rnn.LSTMStateTuple(
#         c=new_enc_state_c, h=new_enc_state_h))

# aa_enc_state = tuple(new_enc_state)

# # with tf.variable_scope("seq2seq_decoder"):
# #     # embedding
# #     dec_cell = bulid_rnn_cell(
# #         'lstm', 20, 2, 0.5)

# #     output_layer = tf.layers.Dense(
# #         6, kernel_initializer=tf.truncated_normal_initializer(0, 0.1))

# #     # train graph
# #     with tf.variable_scope("decode"):
# #         attention_enc_state = dec_cell.zero_state(tf.shape(enc_state)[2], tf.float32)
# #         # dec_input_len = tf.Print(dec_input_len, [dec_input_len])
# #         dec_embedding_input = tf.nn.embedding_lookup(embedding, dec_input)
# #         training_helper = tf.contrib.seq2seq.TrainingHelper(
# #             dec_embedding_input, sequence_length=[3,3], time_major=False)
# #         training_decoder = tf.contrib.seq2seq.BasicDecoder(
# #                 dec_cell, training_helper, attention_enc_state, output_layer)
# #         training_output = tf.contrib.seq2seq.dynamic_decode(training_decoder, impute_finished=False, maximum_iterations=2)[0]
# #         decoder_logits_train = tf.identity(training_output.rnn_output)
# #         decoder_predict_train = tf.argmax(decoder_logits_train, axis=-1, name='decoder_pred_train')

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     a = sess.run(enc_output)
#     b = sess.run(enc_state)
#     c = sess.run(new_enc_state)
#     print(np.array(a).shape)
#     print(np.array(b).shape)
#     print(np.array(c).shape)
#     print(c)

# 	# embed = sess.run(emb_data_input)
# 	# logits = sess.run(training_output)
# 	# print(logits)
# 	# print(np.array(embed).shape)
# 	# #aa = sess.run([decoder_logits_train,decoder_predict_train])
# 	# logits = sess.run(training_output)
# 	# print(logits)