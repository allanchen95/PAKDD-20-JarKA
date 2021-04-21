from __future__ import division
import numpy as np
from transe import *
from train_funcs import *
import tensorflow as tf
import os
import copy
from nltk.translate.bleu_score import sentence_bleu
from data_process import *
from new_attr_model import Seq2SeqModel
import time
import pickle
import gc
import eal_part_txb as eal_part
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--merge', dest='merge_strategy', choices=["multi_view", "score_based", "rank_based"], 
                    default="rank_based", help='select the merge_strategy')
args = parser.parse_args()




# structure model para
ere_file_path = 'data/'
embed_size = 75
learning_rate = 0.01
batch_size = 20000

lambda_1 = 0.01
lambda_2 = 2.0

mu_1 = 0.2
epsilon = 0.9

n = 5
nums_threads = 10
max_train_num = 50

id2_rel1 = {}
with open(ere_file_path + 'rel_ids_1', 'r') as file:
    for line in file:
        token = line.strip('\n').split('\t')
        ids = token[0]
        rels = token[1].strip().split('/property/')[-1]
        id2_rel1[int(ids)] = rels

print("relation1: ",len(id2_rel1))

id2_rel2 = {}
with open(ere_file_path + 'rel_ids_2', 'r') as file:
    for line in file:
        token = line.strip('\n').split('\t')
        ids = token[0]
        rels = token[1].strip().split('/property/')[-1]
        id2_rel2[int(ids)] = rels

print("relation2: ",len(id2_rel2))
# print(relation2[3])



ori_triples1, ori_triples2, new_triples1, new_triples2, seed_sup_ent, val_ent, test_ent, ent_num, rel_num = process_data(ere_file_path)


test_index1, test_index2 = get_test_ents_index(test_ent)

ori_test_index1 = copy.deepcopy(test_index1)
ori_test_index2 = copy.deepcopy(test_index2)

new_triples1_ori = copy.deepcopy(new_triples1)
new_triples2_ori = copy.deepcopy(new_triples2)

#Get counter dict
counter_dict = {}

for pairs in test_ent:
    counter_dict[pairs[0]] = pairs[1]
assert len(counter_dict) == len(test_ent)


print("ori_triples1: ",len(ori_triples1.triples))
print("ori_triples2: ",len(ori_triples2.triples))
print("new_triples1: ",len(new_triples1.triples))
print("new_triples2: ",len(new_triples2.triples))


#generate_related_matrix(new_triples1, new_triples2, ref_ent)

candidate_num = int(len(ori_triples1.ent_list) * (1 - epsilon))

# attribute model para

dict_path = './data/whole_vocab_split'

# test_data_path = '../new_seq_test'
# train_data_path = '../new_seq_train'

num_epoches = 100
rnn_size = 128
num_layers = 2
embedding_size = 100
learning_rate = 0.001
use_attention = True
use_beam_search = True
beam_size = 5
max_gradient_norm = 5.0
use_checkpoint = False

#load_path = './new_ckp111/'
load_path = './pre_train/'
top_k = [1,10,50]

data_folder = './data/'

word_dict = []

with open(dict_path) as f:
    for line in f:
        word = line.strip()
        word_dict.append(word)

word2idx = {w: i for i, w in enumerate(word_dict)}
idx2word = {i: w for i, w in enumerate(word_dict)}
word_size = len(word_dict)


key2id = pickle.load(open(data_folder + 'att2id_fre.p', "rb"))###att-name id:  frequency:

total_en_att = pickle.load(open(data_folder + 'en_ent_att_filter.p', "rb"))

total_zh_att = pickle.load(open(data_folder + 'zh_ent_att_filter.p', "rb"))

en_ent_att = pickle.load(open(data_folder + 'filter_sort_en_att_filter.p', "rb"))

zh_ent_att = pickle.load(open(data_folder + 'filter_sort_zh_att_filter.p', "rb"))

# sup_ent_pairs = read_ref(data_folder + 'sup_pairs')
# print("sup_pairs: ",len(sup_ent_pairs))


sup_train_corpus = get_trainingdata(seed_sup_ent, total_zh_att, total_en_att,key2id)
print("sup_train_corpus: ",len(sup_train_corpus))

train_data, train_corpus = get_batch_data_via_new_corpus(sup_train_corpus, dict_path, 200, None)

epoch_train_data = train_data.createbatchs()

print("Train data has %d batches." % (len(epoch_train_data)))
print("Total train corpus: ",len(train_corpus))



# val_ent, test_ent

val_zh_true_set, val_en_part_set = get_true_val(val_ent, zh_ent_att, en_ent_att, word2idx)




test_zh_true_set, test_en_true_set = get_true_val(test_ent, zh_ent_att, en_ent_att, word2idx)

val_en_true_set = copy.deepcopy(val_en_part_set)
val_en_true_set.extend(test_en_true_set)

ori_test_zh_true_set = copy.deepcopy(test_zh_true_set)
ori_test_en_true_set = copy.deepcopy(test_en_true_set)

ori_test_encode_data, ori_test_encode_batch_data, ori_test_decode_data = prepare_data(ori_test_zh_true_set, ori_test_en_true_set, zh_ent_att, en_ent_att, word2idx,key2id)###


tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
tf_config.gpu_options.allow_growth = True

KGE_graph = tf.Graph()

kge_sess = tf.Session(graph=KGE_graph,config=tf_config)

seq2seq_graph = tf.Graph()

seq_sess = tf.Session(graph=seq2seq_graph,config=tf_config)


#Loading two graphs
with KGE_graph.as_default():
    with kge_sess.as_default():
        with tf.device('/gpu:0'):
            str_model = KGE_model(kge_sess, ent_num, rel_num, len(id2_rel1), len(id2_rel2), ori_triples1.ent_list, ori_triples2.ent_list, val_ent, test_ent, test_index1, test_index2, counter_dict, seed_sup_ent, embed_size, lambda_1, lambda_2, mu_1, 0.01)
            kge_sess.run(tf.global_variables_initializer())




with seq2seq_graph.as_default():
    with seq_sess.as_default():
        with tf.device('/gpu:1'):
            att_model = Seq2SeqModel(sess = seq_sess,
                                rnn_size = rnn_size,
                                num_layers = num_layers,
                                embedding_size = embedding_size,
                                learning_rate = 0.001,
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
                    att_model.saver.restore(seq_sess, ckpt.model_checkpoint_path)
            else:
                print("Create new parameters...")
                seq_sess.run(tf.global_variables_initializer())

set_att_align = set()
aligned_test_pairs = {}
aligned_rel_pairs = {}
# aligned_test_set = seed_sup_ent

# Pre_train att_model for att_seed
print('--------pre_train attribute model-------')
val_en_true_set = copy.deepcopy(val_en_part_set)
val_en_true_set.extend(test_en_true_set)
val_encode_data, val_encode_batch_data, val_decode_data = prepare_data(val_zh_true_set, val_en_true_set,zh_ent_att, en_ent_att, word2idx, key2id)###
pre_train_nmt(att_model, 50, train_data, val_encode_data, val_encode_batch_data, val_decode_data)

print('--------begin iteration-------')
for epoch_num in range(5):
    print("iteration: ", epoch_num + 1)
    print("------------ATTRIBUTE MODEL------------")
    # eal_part.pad_for_nmt(att_model, zh_ent_att, en_ent_att, set(aligned_test_set)|set(seed_sup_ent), word2idx, key2id)
    # add att seed
    aligned_test_set = dict2set(aligned_test_pairs)
    # set_att_align = eal_part.get_align_att_pair(att_model, key2id, zh_ent_att, en_ent_att, set(seed_sup_ent), set_att_align, word2idx)
    set_att_align = eal_part.get_align_att_pair(att_model, key2id, zh_ent_att, en_ent_att, set(aligned_test_set) | set(seed_sup_ent), set_att_align, word2idx)
    key2id = eal_part.change_att2id_with_set_att_align(key2id,set_att_align)

    #attribute model
    newly_train_corpus = get_trainingdata(set(aligned_test_set) | set(seed_sup_ent), total_zh_att, total_en_att, key2id)
    train_data, train_corpus = get_batch_data_via_new_corpus(train_corpus, dict_path, 200, newly_train_corpus)

    val_en_true_set = copy.deepcopy(val_en_part_set)
    val_en_true_set.extend(test_en_true_set)
    val_encode_data, val_encode_batch_data, val_decode_data = prepare_data(val_zh_true_set, val_en_true_set,zh_ent_att, en_ent_att, word2idx, key2id)###
    test_encode_data, test_encode_batch_data, test_decode_data = prepare_data(test_zh_true_set, test_en_true_set, zh_ent_att, en_ent_att, word2idx, key2id)###
    top_ents = train_nmt(att_model, 50, train_data, val_encode_data, val_encode_batch_data, val_decode_data)
    threshold = get_threshold_via_val(top_ents)
    print("----------------Find newly seed-------------")
    att_tmp_dict = find_newly_pairs(att_model, counter_dict,test_zh_true_set, test_en_true_set, test_encode_data, test_encode_batch_data, test_decode_data, threshold, len(test_ent))
    if args.merge_strategy == 'multi_view':
        aligned_test_pairs, final_dict = merge_seed(counter_dict, att_tmp_dict, att_tmp_dict, aligned_test_pairs, strategy="score_based")
        delete_aligned_ents(str_model, final_dict)
        print("before merge: {} / {}".format(len(test_zh_true_set), len(test_en_true_set)))
        test_zh_true_set = list(set(str_model.test_index1) & set(test_zh_true_set))
        test_en_true_set = list(set(str_model.test_index2) & set(test_en_true_set))
        print("after merge: {} / {}".format(len(test_zh_true_set), len(test_en_true_set)))

    #structure model
    print("------------STRUCTURE MODEL------------")
    train_transe(new_triples1, new_triples2, candidate_num, str_model, n, batch_size, max_train_num)
    str_model.test(aligned_test_pairs, ori_test_index1, ori_test_index2)
    test_threshold = str_model.get_threshold_via_val()
    str_tmp_dict = find_potential_alignment(str_model, test_threshold)
    aligned_rel_pairs = find_potential_relations(str_model, id2_rel1, id2_rel2, aligned_rel_pairs)
    # exit()
    if args.merge_strategy == 'score_based':
        aligned_test_pairs, final_dict = merge_seed(counter_dict, str_tmp_dict, att_tmp_dict, aligned_test_pairs, strategy="score_based")
    if args.merge_strategy == 'rank_based' or args.merge_strategy == "multi_view":
        aligned_test_pairs, final_dict = merge_seed(counter_dict, str_tmp_dict, att_tmp_dict, aligned_test_pairs, strategy="rank_based")
    delete_aligned_ents(str_model, final_dict)
    print("before merge: {} / {}".format(len(test_zh_true_set), len(test_en_true_set)))
    test_zh_true_set = list(set(str_model.test_index1) & set(test_zh_true_set))
    test_en_true_set = list(set(str_model.test_index2) & set(test_en_true_set))
    print("after merge: {} / {}".format(len(test_zh_true_set), len(test_en_true_set)))
    #Find newly aligned train instance
    # #attribute model
    # aligned_test_set = dict2set(aligned_test_pairs)
    # newly_train_corpus = get_trainingdata(aligned_test_set, total_zh_att, total_en_att, key2id)
    # train_data, train_corpus = get_batch_data_via_new_corpus(train_corpus, dict_path, 200, newly_train_corpus)
    #structure model
    new_triples1, new_triples2 = get_add_newly_triples(aligned_test_pairs, aligned_rel_pairs, str_model, new_triples1_ori, new_triples2_ori, ori_test_index1, ori_test_index2)




