from __future__ import division
import tensorflow as tf
import numpy as np
import os
from nltk.translate.bleu_score import sentence_bleu
from data_process import *
from new_attr_model import Seq2SeqModel
import time
import pickle
import copy
import gc

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


dict_path = './data/whole_vocab'

test_data_path = '../new_seq_test'
train_data_path = '../new_seq_train'

num_epoches = 100
rnn_size = 128
num_layers = 2
embedding_size = 100
learning_rate = 0.001
use_attention = True
use_beam_search = True
beam_size = 5
max_gradient_norm = 5.0
use_checkpoint = True

load_path = './new_ckp111/'
top_k = [1,10,50]

data_folder = './data/'
aligned_test_pairs = {}



# word_dict = []

# with open(dict_path) as f:
# 	for line in f:
# 		word = line.strip()
# 		word_dict.append(word)

# word2idx = {w: i for i, w in enumerate(word_dict)}
# idx2word = {i: w for i, w in enumerate(word_dict)}
# word_size = len(word_dict)

# s_t = time.time()
# vali_set, encode_data, decode_data= read_validation(data_folder, word2idx)
# print(np.array(encode_data.inputs).shape)
# print(np.array(decode_data.inputs).shape)
# print(time.time() - s_t)

# get test data
word_dict = []

with open('./data/whole_vocab') as f:
	for line in f:
		word = line.strip()
		word_dict.append(word)

word2idx = {w: i for i, w in enumerate(word_dict)}
idx2word = {i: w for i, w in enumerate(word_dict)}
word_size = len(word_dict)




total_en_att = pickle.load(open(data_folder + 'en_ent_att.p', "rb"))

total_zh_att = pickle.load(open(data_folder + 'zh_ent_att.p', "rb"))

en_ent_att = pickle.load(open(data_folder + 'filter_sort_en_att.p', "rb"))

zh_ent_att = pickle.load(open(data_folder + 'filter_sort_zh_att.p', "rb"))

sup_ent_pairs = read_ref(data_folder + 'sup_pairs')
print("sup_pairs: ",len(sup_ent_pairs))


sup_train_corpus = get_trainingdata(sup_ent_pairs, total_zh_att, total_en_att)
print("sup_train_corpus: ",len(sup_train_corpus))

train_data, train_corpus = get_batch_data_via_new_corpus(sup_train_corpus, dict_path, 200, None)

epoch_train_data = train_data.createbatchs()

print("Train data has %d batches." % (len(epoch_train_data)))
print("Total train corpus: ",len(train_corpus))

val_ent_pairs, test_ent_pairs = read_validation(data_folder, word2idx)

counter_dict = get_counter_dict(test_ent_pairs)

assert len(counter_dict) == len(test_ent_pairs)
# print(counter_dict)
# val_encode_data, val_decode_data, _, _ = get_true_val(val_ent_pairs, zh_ent_att, en_ent_att, word2idx)

# test_encode_data, test_decode_data, test_ents1, test_ents2 = get_true_val(test_ent_pairs, zh_ent_att, en_ent_att, word2idx)

val_zh_true_set, val_en_part_set = get_true_val(val_ent_pairs, zh_ent_att, en_ent_att, word2idx)




test_zh_true_set, test_en_true_set = get_true_val(test_ent_pairs, zh_ent_att, en_ent_att, word2idx)

val_en_true_set = copy.deepcopy(val_en_part_set)
val_en_true_set.extend(test_en_true_set)

ori_test_zh_true_set = copy.deepcopy(test_zh_true_set)
ori_test_en_true_set = copy.deepcopy(test_en_true_set)

ori_test_encode_data, ori_test_encode_batch_data, ori_test_decode_data = prepare_data(ori_test_zh_true_set, ori_test_en_true_set, zh_ent_att, en_ent_att, word2idx)




# print("val_zh:",val_zh_true_set)
# print("val_en:",val_en_part_set)
# print("val_t_set:", val_en_true_set)
# print("test_zh:",test_zh_true_set)
# print("test_en:",test_en_true_set) 


seq2seq_graph = tf.Graph()

tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

tf_config.gpu_options.allow_growth = True

sess = tf.Session(graph=seq2seq_graph,config=tf_config)

#Create the graph

with seq2seq_graph.as_default():
	with sess.as_default():
		model = Seq2SeqModel(sess = sess,
							rnn_size = rnn_size, 
							num_layers = num_layers,
							embedding_size = embedding_size, 
							learning_rate = learning_rate,
							word_to_idx = word2idx, 
							use_attention = use_attention,
							beam_search = use_beam_search,
							beam_size = beam_size,
							max_gradient_norm = max_gradient_norm)
		if use_checkpoint:
			ckpt = tf.train.get_checkpoint_state(load_path)
			if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
				print('Reloading model parameters...')
				print(ckpt.model_checkpoint_path)
				model.saver.restore(sess, ckpt.model_checkpoint_path)
		else:
			print("Create new parameters...")
			sess.run(tf.global_variables_initializer())


# print("sleep 60s")

# tf.reset_default_graph()

# time.sleep(60)

		
for epoch_num in range(50):
	print("-----------Epoch:{} / {}-----------".format(epoch_num+1, num_epoches))


	val_en_true_set = copy.deepcopy(val_en_part_set)

	val_en_true_set.extend(test_en_true_set)

	# print("val_zh:",len(val_zh_true_set))
	# print("val_en:",len(val_en_part_set))
	# print("val_t_set:", len(val_en_true_set))
	# print("test_zh:",len(test_zh_true_set))
	# print("test_en:",len(test_en_true_set)) 


	val_encode_data, val_encode_batch_data, val_decode_data = prepare_data(val_zh_true_set, val_en_true_set,zh_ent_att, en_ent_att, word2idx)
	test_encode_data, test_encode_batch_data, test_decode_data = prepare_data(test_zh_true_set, test_en_true_set, zh_ent_att, en_ent_att, word2idx)
	
	
	top_ents = train_nmt(model, 50, train_data, val_encode_data, val_encode_batch_data, val_decode_data)
	
	threshold = get_threshold_via_val(top_ents)
	# print(threshold)

	print("----------------Begin Test-------------")
	test_via_embed(model, counter_dict, ori_test_zh_true_set, ori_test_en_true_set, ori_test_encode_data, ori_test_encode_batch_data, ori_test_decode_data, 500, aligned_test_pairs)

	print("----------------Find newly seed-------------")

	aligned_test_pairs, aligned_test_set, test_zh_true_set, test_en_true_set = find_newly_pairs(model, counter_dict, test_zh_true_set, test_en_true_set, test_encode_data, test_encode_batch_data, test_decode_data, threshold, aligned_test_pairs, len(test_ent_pairs))

	# assert len(test_top_ents) == len(test_en_true_set)


	# aligned_test_pairs = find_newly_pairs(test_top_ents, threshold, aligned_test_pairs, len(test_ent_pairs))

	# print("align:",aligned_test_set)
	# exit(0)

	newly_train_corpus = get_trainingdata(aligned_test_set, total_zh_att, total_en_att)



	train_data, train_corpus = get_batch_data_via_new_corpus(train_corpus, dict_path, 200, newly_train_corpus)

	del val_en_true_set
	del aligned_test_set
	gc.collect()
			

# encode_batch_data, encode_ent_len = createbatchs_for_embed(encode_data, 1500)

# # print(len(encode_batch_data))
# encode_embed = []
# encode_len = []
# print("batch_num:",len(encode_batch_data))
# s_t = time.time()
# for batch_num in range(len(encode_batch_data)):
# 	tmp_embed, tmp_len = model.get_predict_embed(encode_batch_data[batch_num])
# 	encode_embed.extend(tmp_embed)
# 	encode_len.extend(tmp_len)

# print(np.array(encode_embed).shape)

# print(time.time() - s_t)

# # encode_embed, decode_embed = model.test_score(encode_data, decode_data)
# decode_embed = model.get_target_embed(decode_data)
# # print(np.array(decode_embed).shape)

# encode_pad_mat, decode_pad_mat = process_pad_mat(encode_data.ent_len, encode_embed, decode_data.ent_len, decode_embed)
# # print(encode_pad_mat.shape)
# # print(decode_pad_mat.shape)
				
# encode_batch_mat = createbatchs_for_mat(encode_pad_mat, 3)			

# total_mat = []
# for batch_num in range(len(encode_batch_mat)):
# 	tmp_mat = model.eval_sim_mat(encode_batch_mat[batch_num], decode_pad_mat)
# 	# print(tmp_mat)
# 	total_mat.extend(tmp_mat)
# # print(total_mat)
# # print(np.array(total_mat))

