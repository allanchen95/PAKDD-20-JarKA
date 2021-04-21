from __future__ import division
import numpy as np
from transe import *
from train_funcs import *
import tensorflow as tf
import os
import copy


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

ere_file_path = 'dataset/'
embed_size = 75
learning_rate = 0.01
batch_size = 20000

lambda_1 = 0.01
lambda_2 = 2.0

mu_1 = 0.2
epsilon = 0.9

n = 5
nums_threads = 10
max_train_num = 5

# sub_list1 = np.arange(0.7,0.8,0.02)
# sub_list2 = np.arange(0.8,0.9,0.01)
# sub_list3 = np.arange(0.9,0.95,0.001)
# sub_list4 = np.arange(0.95,1,0.0005)

# threshold_list = list(set(sub_list1) | set(sub_list2) | set(sub_list3) | set(sub_list4))
# threshold_list.sort()
# print(threshold_list)
# print(len(threshold_list))


aligned_test_pairs = {}

ori_triples1, ori_triples2, new_triples1, new_triples2, seed_sup_ent, val_ent, test_ent, ent_num, rel_num = process_data(ere_file_path)


test_index1, test_index2 = get_test_ents_index(test_ent)

ori_test_index1 = copy.deepcopy(test_index1)
ori_test_index2 = copy.deepcopy(test_index2)

counter_dict = get_counter_dict(test_ent)


print("ori_triples1: ",len(ori_triples1.triples))
print("ori_triples2: ",len(ori_triples2.triples))
print("new_triples1: ",len(new_triples1.triples))
print("new_triples2: ",len(new_triples2.triples))


#generate_related_matrix(new_triples1, new_triples2, ref_ent)

candidate_num = int(len(ori_triples1.ent_list) * (1 - epsilon))


tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
tf_config.gpu_options.allow_growth = True

KGE_graph = tf.Graph()

kge_sess = tf.Session(graph=KGE_graph,config=tf_config)

with KGE_graph.as_default():
	with tf.Session(graph=KGE_graph,config=tf_config) as sess:
		model = KGE_model(kge_sess, ent_num, rel_num, ori_triples1.ent_list, ori_triples2.ent_list, val_ent, test_ent, test_index1, test_index2, counter_dict, seed_sup_ent, embed_size, lambda_1, lambda_2, mu_1, learning_rate)
		kge_sess.run(tf.global_variables_initializer())
	# var = tf.trainable_variables()
	# print('---')
	# for i in var:
	# 	print(i)
for epoch_num in range(100):
	print("iteration: ", epoch_num + 1)
	# print("ori_ents:{} {}".format(len(ori_triples1.ent_list), len(ori_triples2.ent_list)))
	# print("new_ents:{} {}".format(len(new_triples1.ent_list), len(new_triples2.ent_list)))

	train_transe(new_triples1, new_triples2, candidate_num, model, n, batch_size, max_train_num)
	model.test(aligned_test_pairs, ori_test_index1, ori_test_index2)

	# flag, hit_1, test_threshold = model.get_threshold_via_val(threshold_list)
	
	# # print_threshold_func(flag, hit_1, test_threshold, model)
	test_threshold = model.get_threshold_via_val()

	# aligned_test_pairs = find_potential_alignment(model, test_threshold, aligned_test_pairs)

	aligned_test_pairs, tmp_dict = find_potential_alignment(str_model, test_threshold, aligned_test_pairs)

	# delete aligend ents

	delete_aligned_ents(str_model, tmp_dict)

	# model.test(aligned_test_pairs)

	new_triples1, new_triples2 = get_add_newly_triples(aligned_test_pairs, model, new_triples1, new_triples2, ori_test_index1, ori_test_index2)

	# train_transe_via_newly_alignment(newly_aligned_triples1, newly_aligned_triples2, model, 3, batch_size)
	# newly_aligned_triples1, newly_aligned_triples2 = get_add_newly_triples(aligned_test_pairs, model, new_triples1, new_triples2, ori_test_index1, ori_test_index2)

	# train_transe_via_newly_alignment(newly_aligned_triples1, newly_aligned_triples2, model, 3, batch_size)

	# model.test(aligned_test_pairs, ori_test_index1, ori_test_index2)





		# labeled_align, ents1, ents2 = bootstrapping(model, labeled_align)









