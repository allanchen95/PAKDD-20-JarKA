from __future__ import division
import tensorflow as tf
import pickle
import numpy as np
import random
from attribute_model import Seq2SeqModel
import gc
import time
from collections import defaultdict
from pyjarowinkler import distance
import multiprocessing
import copy
import jieba


def get_model_embed(model, encode_data, encode_batch_data, decode_data, batch_size):
    encode_embed = []
    encode_len = []
    s_t = time.time()
    for batch_num in range(len(encode_batch_data)):
        tmp_embed, tmp_len = model.get_predict_embed(encode_batch_data[batch_num])
        encode_embed.extend(tmp_embed)
        encode_len.extend(tmp_len)

        del tmp_embed
        del tmp_len
        

    print("get_predict_embed time_cost: ",round(time.time() - s_t,6))


    # encode_embed, decode_embed = model.test_score(encode_data, decode_data)
    decode_embed = model.get_target_embed(decode_data)
    # print(np.array(decode_embed).shape)

    encode_pad_mat, decode_pad_mat = process_pad_mat(encode_data.ent_len, encode_embed, decode_data.ent_len, decode_embed)
    print("encode_pad_mat shape: ",encode_pad_mat.shape)
    print("decode_pad_mat shape: ",decode_pad_mat.shape)
                    
    encode_batch_mat = createbatchs_for_mat(encode_pad_mat, batch_size)
    encode_batch_key = createbatchs_for_mat(encode_data.key_list, batch_size)    

    assert len(encode_batch_key) == len(encode_batch_mat)

    total_mat = []
    print("eval sim mat batch_num: ",len(encode_batch_mat))
    for batch_num in range(len(encode_batch_mat)):
        batch_mat = encode_batch_mat[batch_num]
        batch_key = encode_batch_key[batch_num]

        assert len(batch_mat) == len(batch_mat)

        tmp_mat = model.eval_sim_mat(batch_mat, decode_pad_mat, batch_key, decode_data.key_list)
        # print(tmp_mat)    
        total_mat.extend(tmp_mat)
        del tmp_mat


    print("entities sim mat: ",np.array(total_mat).shape)
    del decode_embed
    del encode_embed
    del encode_len

    del encode_pad_mat
    del decode_pad_mat

    del encode_batch_mat
    del encode_batch_key
    # print(np.array(total_key))
    gc.collect()

    return np.array(total_mat)


def test_via_embed(model, counter_dict, zh_set, en_set, encode_data, encode_batch_data, decode_data, batch_size, aligned_test_pairs = {}):

    top_k = [1,10,50]
    top_k_metric = np.array([0 for k in top_k])
    top_k_metric1 = np.array([0 for k in top_k])

 
    sim_mat = get_model_embed(model, encode_data, encode_batch_data, decode_data, batch_size)


    for i in range(len(sim_mat)):

        rank = np.argsort(-sim_mat[i,:])

        true_index = np.where(rank == i)[0][0]



        for k in range(len(top_k)):
            if true_index < top_k[k]:
                top_k_metric[k] += 1

        if (len(aligned_test_pairs) > 0 ) and aligned_test_pairs.get(zh_set[i])!=None:
            ent = aligned_test_pairs.get(zh_set[i])
            check_set = set(en_set)
            if(ent in check_set):
                index = en_set.index(ent)
                # print("{} / {}".format(ent, index))
                sim_mat[i][index] += 100
                rank = np.argsort(-sim_mat[i,:])
                true_index = np.where(rank == i)[0][0]
                for k in range(len(top_k)):
                    if true_index < top_k[k]:
                        top_k_metric1[k] += 1
            else:
                for k in range(len(top_k)):
                    if true_index < top_k[k]:
                        top_k_metric1[k] += 1

        else:
            for k in range(len(top_k)):
                if true_index < top_k[k]:
                    top_k_metric1[k] += 1


    print("score_matrix_len: ",len(sim_mat))
    ratio_top_k = np.array([0 for i in top_k], dtype = np.float32)
    ratio_top_k1= np.array([0 for i in top_k], dtype = np.float32)
    for i in range(len(ratio_top_k)):
        ratio_top_k[i] = round(top_k_metric[i] / len(sim_mat), 4)    

    for i in range(len(ratio_top_k)):
        ratio_top_k1[i] = round(top_k_metric1[i] / len(sim_mat), 4)    

    
    print("without aligned test_hits@: {} = {}".format(top_k, ratio_top_k))
    print("with aligned test_hits@: {} = {}".format(top_k, ratio_top_k1))

    del sim_mat

    gc.collect()


def val_via_embed(model, encode_data, encode_batch_data, decode_data, batch_size):
    
 
    sim_mat = get_model_embed(model, encode_data, encode_batch_data, decode_data, batch_size)

    top_k_metric, top_ents = eval_top_k(sim_mat)



    del sim_mat

    gc.collect()

    return top_k_metric, top_ents


def eval_top_k(score_matrix):

    top_k = [1,10,50]
    top_k_metric = np.array([0 for k in top_k])
    top_ents = set()

    for i in range(len(score_matrix)):
        rank = np.argsort(-score_matrix[i,:])

        true_index = np.where(rank == i)[0][0]

        

        score = score_matrix[i][rank[0]]

        top_ents.add((i, rank[0], score))


        for k in range(len(top_k)):
            if true_index < top_k[k]:
                top_k_metric[k] += 1

    ratio_top_k = np.array([0 for i in top_k], dtype = np.float32)
    print("score_matrix_len: ",len(score_matrix))
    for i in range(len(ratio_top_k)):
        ratio_top_k[i] = round(top_k_metric[i] / len(score_matrix), 4)    
    
    del score_matrix
    del top_k_metric
    gc.collect()

    return ratio_top_k, top_ents


    # print(total_mat)
    # print(np.array(total_mat))



def process_pad_mat(en_ent_len, encode_embed, de_ent_len, decode_embed):
    
    assert (np.sum(en_ent_len) == len(encode_embed)) and (np.sum(de_ent_len) == len(decode_embed))

    en_att_matrix = np.zeros([len(en_ent_len), 20, 100], dtype = np.float16)
    count = 0
    for ent_index in range(len(en_ent_len)):

        en_length = en_ent_len[ent_index]
        #print(en_length)

        en_att_matrix[ent_index][0:en_length] = encode_embed[count:count + en_length]
        count += en_length
    

    de_att_matrix = np.zeros([len(de_ent_len), 20, 100], dtype = np.float16)
    count = 0
    #print(np.sum(de_ent_len))
    for ent_index in range(len(de_ent_len)):
        en_length = de_ent_len[ent_index]
        #print(en_length)

        de_att_matrix[ent_index][0:en_length] = decode_embed[count:count + en_length]
        count += en_length
    del encode_embed
    del decode_embed
    gc.collect()

    return en_att_matrix, de_att_matrix


def read_ref(file_name):
    ref_pairs = set()
    with open(file_name,'r') as file:
        for line in file:
            token = line.strip('\n').split('\t')
            assert len(token) == 2
            ref_pairs.add((int(token[0]), int(token[1])))

    return ref_pairs

def get_batch_data_via_new_corpus(old_corpus, dict_path, batch_size, new_corpus = None):

    if (new_corpus != None) and (len(new_corpus) != 0):
        total_corpus = old_corpus | new_corpus 
    else:
        total_corpus = old_corpus

    print("Total_corpus: ",len(total_corpus))

    train_data = data_process(total_corpus, dict_path, batch_size)

    return train_data, total_corpus



def revise_alignment(c_dict, top_ents, aligned_pairs):
    reverse_aligned_pairs = {}
    new_aligened_pairs = {}
    check_dict = {}
    for i,j,k in top_ents:
        key = str(i) + '/' + str(j)
        check_dict[key] = k


    for i, j in aligned_pairs.items():


        ent_j = reverse_aligned_pairs.get(j[0] ,set())
        ent_j.add((i,j[1]))
        reverse_aligned_pairs[j[0]] = ent_j

    for j, ent_j in reverse_aligned_pairs.items():
        if(len(ent_j) == 1):
            for i in ent_j:
                key = i[0]
                value = (j, i[1])
                new_aligened_pairs[key] = value 
        else:
            max_i = -1
            max_sim = -100 
            for i in ent_j:
                key = str(i[0]) + '/' + str(j)
                if check_dict.get(key) != None:
                    if(check_dict[key] > max_sim):
                        max_sim = check_dict[key]
                        max_i = i

            new_aligened_pairs[max_i[0]] = (j, max_i[1])
    check_alignment(c_dict, new_aligened_pairs, "After revised: ")

    return new_aligened_pairs


def delete_aligned_pairs(tmp_dict, test_ents1, test_ents2):
    m_test_ents1 = copy.deepcopy(test_ents1)
    m_test_ents2 = copy.deepcopy(test_ents2)

    for i, j in tmp_dict.items():
        # align_set.add((test_ents1[i],test_ents2[j]))
        # print("pop:{}/ {}".format(i, j))
        m_test_ents1.remove(i)
        m_test_ents2.remove(j)



    return m_test_ents1, m_test_ents2


def dict2set(aligned_test_pairs):
    
    # align_set = set()
    # for i, j in aligned_test_pairs.items():
    #     align_set.add((test_ents1[i],test_ents2[j]))
    align_set = set()
    for i, j in aligned_test_pairs.items():
        align_set.add((i,j))


    assert len(align_set) == len(aligned_test_pairs)


    return align_set

# def dict2set(aligned_test_pairs, tmp_dict, test_ents1, test_ents2):
    
#     # align_set = set()
#     # for i, j in aligned_test_pairs.items():
#     #     align_set.add((test_ents1[i],test_ents2[j]))
#     align_set = set()
#     for i, j in aligned_test_pairs.items():
#         align_set.add((i,j))


#     assert len(align_set) == len(aligned_test_pairs)

#     m_test_ents1 = copy.deepcopy(test_ents1)
#     m_test_ents2 = copy.deepcopy(test_ents2)

#     for i, j in tmp_dict.items():
#         # align_set.add((test_ents1[i],test_ents2[j]))
#         # print("pop:{}/ {}".format(i, j))
#         m_test_ents1.remove(i)
#         m_test_ents2.remove(j)



#     return align_set, m_test_ents1, m_test_ents2


def find_newly_pairs(model, c_dict, zh_set, en_set, encode_data, encode_batch_data, decode_data, threshold, len_n):
    top_k = [1,10,50]

    # top_k_metric, top_ents = test_via_embed(model, encode_data, encode_batch_data, decode_data, batch_size)

    sim_mat = get_model_embed(model, encode_data, encode_batch_data, decode_data, 500)

    top_ents = set()
    tmp_dict = {}
    # tmp_dict1 = {}

    for i in range(len(sim_mat)):
        rank = np.argsort(-sim_mat[i,:])

        score = sim_mat[i][rank[0]]

        top_ents.add((zh_set[i], en_set[rank[0]], score))


    tmp_dict = find_alignment_via_sim(c_dict,top_ents, threshold)

    tmp_dict = revise_alignment(c_dict, top_ents, tmp_dict)


    # num = 0

    # for key, value in tmp_dict.items():
    #     tmp_dict1[key] = value[0]

    # assert len(tmp_dict) == len(tmp_dict1)

    # aligned_test_pairs = {**aligned_test_pairs, **tmp_dict1}

    # for x, y in aligned_test_pairs.items():
    #     if c_dict[x] == y:
    #         num += 1
    # print("whole aligned_pairs, right alignment: {}/{}={:.3f}".format(num, len(aligned_test_pairs), num / len(aligned_test_pairs)))


    # precision = round(num / len(aligned_test_pairs), 5)
    # recall = round(num / len_n, 5)
    # f1 = round(2 * precision * recall / (precision + recall), 6)
    # print("Whole precision={}, recall={}, f1={}".format(precision, recall, f1))


    # aligned_test_set = dict2set(aligned_test_pairs)
    # test_zh_true_set, test_en_true_set = delete_aligned_pairs(tmp_dict, zh_set, en_set)

    del sim_mat
    del top_ents
    # del tmp_dict1
    gc.collect()

    # return aligned_test_pairs, aligned_test_set, test_zh_true_set, test_en_true_set
    return tmp_dict


def find_alignment_via_sim(c_dict, top_ents, threshold):


    total_num = 0
    hit_num = 0
    len_n = len(top_ents)
    top_ents_list = sorted(top_ents, key=lambda x:x[2], reverse=True)
    tmp_dict = {}

    # for pairs in top_ents_list:
    #     if (pairs[2] >= threshold):
    #         total_num += 1
    #         # aligned_test_pairs[pairs[0]] = pairs[1]
    #         tmp_dict[pairs[0]] = pairs[1]
    #         # if c_dict[pairs[0]] == pairs[1]:
    #         #     hit_num += 1
    #     else:
    #         break
    for i in range(len(top_ents_list)):
        pairs = top_ents_list[i]
        if(pairs[2] >= threshold):
            total_num += 1
            value = (pairs[1], i)
            tmp_dict[pairs[0]] = value
        else:
            break

    if(total_num > 0):
        check_alignment(c_dict, tmp_dict, "After test, before revised: ")
        return tmp_dict
    else:
        print("No more test seed proposed!")
        # exit(0)
        return {}

def check_alignment(c_dict, tmp_dict, context = ""):
    # if aligned_pairs is None or len(aligned_pairs) == 0:
    #     print("{}, Empty aligned pairs".format(context))
    #     return

    if tmp_dict is None or len(tmp_dict) == 0:
        print("{}, Empty aligned pairs".format(context))
        
        return
    num = 0

    for x, y in tmp_dict.items():
        if c_dict[x] == y[0]:
            num += 1
    print("This iteration {}, right alignment: {}/{}={:.3f}".format(context, num, len(tmp_dict), num / len(tmp_dict)))



def get_trainingdata(data_pairs, zh_att_triples, en_att_triples,key2id):


    def is_digit(att_value):
        modi_value = ''.join(att_value.split())
        try:
            float(modi_value)
            return True
        except:
            return False

    train_pairs = set()

    for pair in data_pairs:
        zh = str(pair[0])
        en = str(pair[1])


        if zh in zh_att_triples:
            if en in en_att_triples:
                count_totoal = 0
                for zh_att in zh_att_triples[zh]:
                    for en_att in en_att_triples[en]:
                        flag_temp=False
                        if zh_att==en_att:
                            flag_temp=True
                        if zh_att in key2id and en_att in key2id:
                            if key2id[zh_att]['id']==key2id[en_att]['id']:
                                flag_temp=True
                        if flag_temp==True:
                            count_totoal = count_totoal + 1
                            # if count_totoal%10000 == 0:
                            #     print ("count_totoal: ",count_totoal)
                            #print ("att: ",zh_att)
                            zh_value = zh_att_triples[zh][zh_att]
                            en_value = en_att_triples[en][en_att]

                            zh_digit_val = set()
                            zh_nodigit_val = set()

                            en_digit_val = set()
                            en_nodigit_val = set()

                            for val in zh_value:
                                m_val = ''.join(val.split())
                                if is_digit(m_val):
                                    zh_digit_val.add(val)
                                else:
                                    zh_nodigit_val.add(val)

                            for val in en_value:
                                m_val = ''.join(val.split())
                                if is_digit(m_val):
                                    en_digit_val.add(val)
                                else:
                                    en_nodigit_val.add(val)

                            digit_sort = set()
                            if (len(zh_digit_val) > 0) & (len(en_digit_val) > 0):
                                for zh_digit_value in zh_digit_val:
                                    for en_digit_value in en_digit_val:
                                        sim = distance.get_jaro_distance(zh_digit_value, en_digit_value, winkler=True, scaling=0.1)
                                        digit_sort.add((sim, zh_digit_value, en_digit_value))
                                sort_list = sorted(digit_sort, key=lambda x: x[0], reverse=True)
                                #print ("sort_list: ", sort_list).
                                left_str = ''
                                right_str = ''
                                for i in sort_list:
                                    if i[0] > 0.9:
                                        left_str = left_str + ' ' + str(i[1])
                                        right_str = right_str + ' ' + str(i[2])

                                if (len(left_str.strip()) > 0) & (len(right_str.strip()) > 0):

                                    train_pairs.add((left_str.strip(), right_str.strip()))

                            if(len(zh_nodigit_val) > 0) & (len(en_nodigit_val) > 0):
                                left_str = ''
                                right_str = ''

                                for zh_nodigit_value in zh_nodigit_val:
                                    if(len(zh_nodigit_value) > 0):
                                        # re_zh = ''
                                        # for i in zh_nodigit_value.split():
                                        #     split_text = jieba.lcut(i, cut_all=False, HMM=True)
                                        #     corpus = ' '.join(split_text)
                                        #     re_zh = re_zh + corpus + ' '
                                        # # left_str = left_str + ' ' + str(zh_nodigit_value)
                                        # left_str = left_str + ' ' + str(re_zh.strip())
                                        left_str = left_str + ' ' + str(zh_nodigit_value)

                                for en_nodigit_value in en_nodigit_val:
                                    if(len(en_nodigit_value) > 0):
                                        right_str = right_str + ' ' + str(en_nodigit_value)

                                if (len(left_str.strip()) > 0) & (len(right_str.strip()) > 0):
                                    train_pairs.add((left_str.strip(), right_str.strip()))

            else:
                 print ("This en_ent is not in en_triples: ",en)
        else:
            print ("This zh_ent is not in zh_triples: ",zh)

    # print(len(train_pairs))
    # print(list(train_pairs)[:20])


    return train_pairs

def read_validation(folder, word2idx):
    ref_ent_pairs = read_ref(folder + 'ref_pairs')
    print("align pairs: ",len(ref_ent_pairs))

    ref_len = len(ref_ent_pairs)

    ref_ent_pairs_list = list(ref_ent_pairs)


    val_index = random.sample(range(ref_len), 1000)
    #val_index = range(20)

    val_ent_pairs_list = [ref_ent_pairs_list[i] for i in val_index]

    print("val_pairs: ",len(val_ent_pairs_list))

    val_ent_pairs = set(val_ent_pairs_list)



    #test_index = random.sample(range(ref_len), 2000)

    # test_index = range(20,100)

    # test_ent_pairs_list = [ref_ent_pairs_list[i] for i in test_index]

    # test_ent_pairs = set(test_ent_pairs_list)

    
    test_ent_pairs = ref_ent_pairs - val_ent_pairs

    #test_ent_pairs = ref_ent_pairs

    print("test_pairs: ",len(test_ent_pairs))





    # val_encode_data, val_decode_data, _, _ = get_true_val(val_ent_pairs, folder, word2idx)

    # test_encode_data, test_decode_data, test_ents1, test_ents2 = get_true_val(test_ent_pairs, folder, word2idx)

    # print("ent_num: {} zh att value num: {}".format(len(val_ent_pairs), np.array(val_encode_data.inputs).shape))
    # print("ent_num: {} en att value num: {}".format(len(val_ent_pairs), np.array(val_decode_data.inputs).shape))


    # print("ent_num: {} zh att value num: {}".format(len(test_ent_pairs), np.array(test_encode_data.inputs).shape))
    # print("ent_num: {} en att value num: {}".format(len(test_ent_pairs), np.array(test_decode_data.inputs).shape))
    return val_ent_pairs, test_ent_pairs
    # return val_ent_pairs, val_encode_data, val_decode_data, test_ent_pairs, test_encode_data, test_decode_data, test_ents1, test_ents2
    # encode_ent_att_mat, encode_total_length, encode_ent_length, decode_ent_att_mat, decode_total_length, decode_ent_length = get_true_val(val_ent_pairs, folder, word2idx)
    # return val_ent_pairs,encode_ent_att_mat, encode_total_length, encode_ent_length, decode_ent_att_mat, decode_total_length, decode_ent_length

def get_counter_dict(test_ent_pairs):
    c_dict = {}

    for pairs in test_ent_pairs:
        c_dict[pairs[0]] = pairs[1]

    return c_dict


def prepare_data(test_zh_true_set, test_en_true_set, zh_ent_att, en_ent_att, word2idx,key2id):


    test_encode_data, test_decode_data = get_padding_data(test_zh_true_set, test_en_true_set, en_ent_att, zh_ent_att, word2idx,key2id)




    print("test_ent_num: {} zh att value num: {}".format(len(test_zh_true_set), np.array(test_encode_data.inputs).shape))
    print("test_ent_num: {} en att value num: {}".format(len(test_en_true_set), np.array(test_decode_data.inputs).shape))
    # val_pairs, val_encode_data, val_decode_data, test_pairs, test_encode_data, test_decode_data, test_ent1, test_ent2 = read_validation(data_folder, word2idx)

    # val_train_corpus = get_trainingdata(val_ent_pairs, total_zh_att, total_en_att)
    # print("val_train_corpus: ",len(val_train_corpus))

    # train_corpus = sup_train_corpus | val_train_corpus

    # print("train_corpus: ",len(train_corpus))

    # train_data, train_corpus = get_batch_data_via_new_corpus(sup_train_corpus, dict_path, 200, val_train_corpus)


    test_encode_batch_data = createbatchs_for_embed(test_encode_data, 1000)

    print("encode batch_num: ",len(test_encode_batch_data))

    return test_encode_data, test_encode_batch_data, test_decode_data




def get_true_val(ent_pair, zh_ent_att, en_ent_att, word2idx):
    # en_ent_att = pickle.load(open(folder + 'filter_sort_en_att.p', "rb"))

    # zh_ent_att = pickle.load(open(folder + 'filter_sort_zh_att.p', "rb"))

    count = 0
    zh_true_set = []
    en_true_set = []
    for pairs in ent_pair:
        zh_ent = pairs[0]
        en_ent = pairs[1]
        #print(str(zh_ent) + '--' + str(en_ent))
        if (zh_ent_att.get(str(zh_ent)) != None) and (en_ent_att.get(str(en_ent)) != None):
            zh_true_set.append((zh_ent))
            en_true_set.append((en_ent))

    print("vaild/test set:", len(zh_true_set))
    # print("zh_true_set: ",zh_true_set)
    # print("en_true_set: ",en_true_set)
    # exit(0)
    # encode_data, decode_data = get_padding_data(zh_true_set, en_true_set, en_ent_att, zh_ent_att, word2idx)

    # return encode_data, decode_data, zh_true_set, en_true_set

    return zh_true_set, en_true_set
    # encode_ent_att_mat, encode_total_length, encode_ent_length, decode_ent_att_mat, decode_total_length, decode_ent_length = get_padding_data(zh_true_set, en_true_set, en_ent_att, zh_ent_att, word2idx)

    # return encode_ent_att_mat, encode_total_length, encode_ent_length, decode_ent_att_mat, decode_total_length, decode_ent_length


def get_padding_data(zh_true_set, en_true_set, en_ent_att, zh_ent_att, word2idx,key2id):


    encode_data = eval_batchs()
    decode_data = eval_batchs()

    zh_ent_att_matrix, zh_total_length, zh_ent_length, encode_key_list = process_pad(zh_true_set, zh_ent_att, word2idx,key2id)

    en_ent_att_matrix, en_total_length, en_ent_length, decode_key_list = process_pad(en_true_set, en_ent_att, word2idx,key2id)

    encode_data.inputs = zh_ent_att_matrix
    encode_data.len = zh_total_length
    encode_data.ent_len = zh_ent_length
    encode_data.key_list = encode_key_list

    decode_data.inputs = en_ent_att_matrix
    decode_data.len = en_total_length
    decode_data.ent_len = en_ent_length
    decode_data.key_list = decode_key_list


    return encode_data, decode_data
    # return zh_ent_att_matrix, zh_total_length, zh_ent_length, en_ent_att_matrix, en_total_length, en_ent_length



def process_pad(true_set, ent_att, word2idx,key2id):


    max_att_index = len(key2id)
    pad_ent_len = 20
    max_att_words = 20
    max_att_num = 20

    ent_att_matrix = []
    total_length = []
    ent_length = []
    key_list = []
    unk_int = word2idx['<UNK>']
    # print(unk_int)
    # self.unk_int = self.word2id['<UNK>']
    pad_int = word2idx['<PAD>']
    for i in range(len(true_set)):
        ent = str(true_set[i])
        att_list =  ent_att[ent]
        # per_ent_mat = np.zeros([pad_ent_len, max_att_words], dtype=np.int32)
        ent_length.append(len(att_list))
        tmp_key = []
        for key, value_list in att_list.items():
            if (key2id.get(key) == None):
                print("No match attr!")
                exit(0)
            
            tmp_key.append(key2id.get(key)['id'])

            att = ''
            for value in value_list:
                att += value + ' '
            att = att.strip().split()[:20]
            att2id = list(map(lambda words : word2idx.get(words, unk_int), att))
            total_length.append(len(att2id))
            att2id = att2id + [pad_int] * (max_att_words - len(att2id))
            # print(att)
            ent_att_matrix.append(att2id)
        assert len(tmp_key) == len(att_list)
        #tmp_key = tmp_key + [0]*(20 - len(tmp_key))
        tmp_key_onehot = np.zeros((max_att_num, max_att_index+1))
        tmp_key_onehot[np.arange(len(tmp_key)), tmp_key] = 1

        # print(tmp_key_onehot.shape)
        # print(np.sum(tmp_key_onehot,1))
        # exit(0)
        key_list.append(tmp_key_onehot)



    return ent_att_matrix, total_length, ent_length, np.array(key_list)


def createbatchs_for_mat(pad_mat, batch_size):
    total_data = []
    data_len = len(pad_mat)
    # print(eval_batchs.len)


    def batch_iterator():
        for i in range(0, data_len, batch_size):
            yield pad_mat[i:min(i + batch_size, data_len)] 



    for samples in batch_iterator():
        # print(samples.shape)
        # batch_data = self.process_batchs(samples)
        total_data.append(samples)

    return total_data


def createbatchs_for_embed(eval_batchs, batch_size):
    total_data = []
    data_len = len(eval_batchs.inputs)
    # print(eval_batchs.len)


    def batch_iterator():
        for i in range(0, data_len, batch_size):
            yield (eval_batchs.inputs[i:min(i + batch_size, data_len)], eval_batchs.len[i:min(i + batch_size, data_len)]) 



    for data, length in batch_iterator():
        mini_ba = mini_batchs()
        mini_ba.inputs = data
        mini_ba.len = length
        # batch_data = self.process_batchs(samples)
        total_data.append(mini_ba)
    return total_data

    # print(total_length)
    # print(zh_ent_att_matrix)

    # print(ent_length)
    # print(len(ent_length))
    # print(np.sum(ent_length))

# def get_padding_data(zh_true_set, en_true_set, en_ent_att, zh_ent_att, word2idx):
    
#     pad_ent_len = 20
#     max_att_words = 10
#     # zh_ent_att_matrix = np.zeros([len(zh_true_set), pad_ent_len, max_att_words], dtype=np.int32)
#     # zh_ent_att_len = np.zeros([len(zh_true_set), pad_ent_len], dtype=np.int32)
#     zh_ent_att_matrix = []
#     zh_true_set = list(zh_true_set)

#     for i in range(len(zh_true_set)):
#         zh_ent = zh_true_set[i]
#         zh_att_list =  zh_ent_att[zh_ent]
#         tmp_list = []
#         tmp_len = []
#         # per_ent_mat = np.zeros([pad_ent_len, max_att_words], dtype=np.int32)
#         for key, value_list in zh_att_list.items():            
#             print(key)
#             att = ''
#             for value in value_list:
#                 att += value + ' '
#             att = att.strip().split()[:10]
#             att2id = list(map(lambda words : word2idx.get(words, 1), att))
#             tmp_len.append(len(att2id))
#             tmp_list.append(att2id)
#             # print(att)
#             # print(att2id)
#             # print(zh_ent_att_len)
#             # print(tmp_list)
        
#         # tmp_max_len = np.max(tmp_len)
#         print(tmp_list)
#         for j in range(len(tmp_list)):
#             tmp_list[j] = tmp_list[j] + [0] * (max_att_words - len(tmp_list[j]))
#         print(tmp_list)
#         print(tmp_len)
#         zh_ent_att_len[i][0:0 + len(tmp_len)] = tmp_len
#         print(zh_ent_att_len)

#         zh_ent_att_matrix[i][0:0+len(tmp_list), 0: 0 + tmp_max_len] = tmp_list
#         print(zh_ent_att_matrix)
#         exit(0)
class mini_batchs():
    def __init__(self):
        self.inputs = []
        self.len = []

        
class eval_batchs():
    def __init__(self):
        self.inputs = []
        self.len = []
        self.ent_len = []
        self.key_list = []



class data_batchs():
    def __init__(self):
        self.encode_inputs = []
        self.encode_len = []

        self.decode_inputs_in = []
        self.decode_inputs_out = []
        self.decode_len = []

class data_process():
    def __init__(self, train_corpus, dict_path, batch_size):
        self.data_path = './data/zh_en_des'

        self.train_corpus = train_corpus
        self.dict_path = dict_path
        self.batch_size = batch_size

        self.word2id, self.id2word, self.vocab_size = self.get_word_dict()
        
        self.eos_int = self.word2id['<EOS>']
        self.go_int = self.word2id['<GO>']
        self.pad_int = self.word2id['<PAD>']
        self.unk_int = self.word2id['<UNK>']

        # print(self.eos_int)
        # print(self.go_int)
        # print(self.pad_int)
        # print(self.unk_int)
        # exit(0)



        self.train_data = self.load_train_data()
        self.data_len = len(self.train_data)

    def get_word_dict(self):
        word_dict = []

        with open(self.dict_path) as f:
            for line in f:
                word = line.strip()
                word_dict.append(word)

        word2idx = {w: i for i, w in enumerate(word_dict)}
        idx2word = {i: w for i, w in enumerate(word_dict)}
        word_size = len(word_dict)

        return word2idx, idx2word, word_size


    def load_train_data(self):
        train_data = []
        print("loadding train data ")
        length = []

        for pairs in self.train_corpus:
            zh_att = pairs[0].split()
            en_att = pairs[1].split()
            #length.append((len(zh_att)))
            #length.append((len(en_att)))
            train_data.append((' '.join(zh_att[:20]), ' '.join(en_att[:20])))
        
        # For pre_train:
        # with open(self.data_path) as f:

        #     for line in f:
        #         token = line.strip().split("\t")
        #         zh_att = token[0].split()
        #         en_att = token[1].split()
        #         #length.append((len(zh_att)))
        #         #length.append((len(en_att)))
        #         train_data.append((' '.join(zh_att[:20]), ' '.join(en_att[:20])))
        #         #train_data.append((token[0][:160], token[1][:160]))
        #         #train_data.append((token[0][:30], token[1][:30]))
                

        print("loadding %d data" % len(train_data))

        
        return train_data

    def process_batchs(self, samples):
        batch_data = data_batchs()

        samples_id = list(map(lambda pairs : (list(map(lambda x:(self.word2id.get(x, self.unk_int)),pairs[0].split())),list(map(lambda x: (self.word2id.get(x, self.unk_int)),pairs[1].split()))) , samples))
        

        #print(samples)
        #print(np.array(samples).shape)
        #print(samples[0][1])
        
        batch_data.encode_len = [len(pairs[0]) for pairs in samples_id]
        #print(batch_data.encode_len)
        batch_data.decode_len = [len(pairs[1])+1 for pairs in samples_id]

        max_encode_len = max(batch_data.encode_len)
        max_decode_len = max(batch_data.decode_len)


        for pairs in samples_id:
            #print("source : ", pairs[0])
            #source = list(reversed(pairs[0]))
            source = pairs[0]
            #print("re source : ", source)
            pad = [self.pad_int]*(max_encode_len - len(source))
            batch_data.encode_inputs.append(source + pad)
            #print(batch_data.encode_inputs)

            target = pairs[1]
            eos_int = [self.eos_int]
            go_int = [self.go_int]
            pad = [self.pad_int] * (max_decode_len - len(target)-1)
            batch_data.decode_inputs_in.append(go_int + target + pad)
            batch_data.decode_inputs_out.append(target + eos_int + pad)

        return batch_data

    def createbatchs(self):
        random.shuffle(self.train_data)
        total_data = []
        def batch_iterator():
            for i in range(0, self.data_len, self.batch_size):
                yield self.train_data[i:min(i + self.batch_size, self.data_len)] 

        for samples in batch_iterator():
            batch_data = self.process_batchs(samples)
            total_data.append(batch_data)
        return total_data

def train_nmt_wo_fix(model, val_encode_data, val_encode_batch_data, val_decode_data):

    cur_step = 0
    display_step = 100
    test_step = 300
    min_epoch_loss = 100
    max_pre = 0
    top_k = [1,10,50]
    final_ents_pairs = set()

    top_k_metric, top_ents = val_via_embed(model, val_encode_data, val_encode_batch_data, val_decode_data, 500)

    print("val_hits@: {} = {}".format(top_k, top_k_metric))

    final_ents_pairs = top_ents

    if (top_k_metric[0] <= max_pre):
        print("Max validation hit@1 - now: {} / max: {}".format(top_k_metric[0], max_pre))

    else:
        max_pre = top_k_metric[0]

    return final_ents_pairs


def pre_train_nmt(model, max_epoch_num, train_data, val_encode_data, val_encode_batch_data, val_decode_data):

    cur_step = 0
    display_step = 100
    test_step = 300
    min_epoch_loss = 100
    max_pre = 0
    top_k = [1,10,50]
    final_ents_pairs = set()


    for epoch_num in range(max_epoch_num):
        print("-----------Train-Epoch:{} / {}-----------".format(epoch_num+1, max_epoch_num))

        epoch_sum_loss = 0
        epoch_train_data = train_data.createbatchs()

        for batch in epoch_train_data:
            cur_step = cur_step + 1

            loss, model_predict = model.train(batch)

            epoch_sum_loss = epoch_sum_loss + round(loss,3)

            if(cur_step > 0) & (cur_step % display_step == 0):
                print("Cur_step: {} Loss: {}".format(cur_step, loss))

        epoch_avg_loss = round(epoch_sum_loss / len(epoch_train_data) ,6)

        print("Total avg loss: {} ".format(epoch_avg_loss))

        if(epoch_num > 0) and (epoch_num % 4 == 0):
        # if (epoch_num == 0):
            

            top_k_metric, top_ents = val_via_embed(model, val_encode_data, val_encode_batch_data, val_decode_data, 500)

            print("val_hits@: {} = {}".format(top_k, top_k_metric))

            final_ents_pairs = top_ents

            if (top_k_metric[0] <= max_pre):
                print("Max validation hit@1 - now: {} / max: {}".format(top_k_metric[0], max_pre))
                break

            else:
                max_pre = top_k_metric[0]

    return final_ents_pairs

def train_nmt(model, max_epoch_num, train_data, val_encode_data, val_encode_batch_data, val_decode_data):

    cur_step = 0
    display_step = 100
    test_step = 300
    min_epoch_loss = 100
    max_pre = 0
    top_k = [1,10,50]
    final_ents_pairs = set()


    for epoch_num in range(max_epoch_num):
        print("-----------Train-Epoch:{} / {}-----------".format(epoch_num+1, max_epoch_num))


                    

        # top_k_metric, top_ents = val_via_embed(model, val_encode_data, val_encode_batch_data, val_decode_data,500)
        # # sim_mat = get_model_embed(model, val_encode_data, val_encode_batch_data, val_decode_data,1000)

        # # top_k_metric, top_ents = eval_top_k(sim_mat)

        # top_ents_list = sorted(top_ents, key=lambda x:x[2], reverse=True) 

        # # ratio_top_k = process_top_k(top_k_metric, None)

        # print("val_hits@: {} = {}".format(top_k, top_k_metric))
        # final_ents_pairs = top_ents




        # exit(0)


        epoch_sum_loss = 0
        epoch_train_data = train_data.createbatchs()

        for batch in epoch_train_data:
            cur_step = cur_step + 1

            loss, model_predict = model.train(batch)

            epoch_sum_loss = epoch_sum_loss + round(loss,3)

            if(cur_step > 0) & (cur_step % display_step == 0):
                print("Cur_step: {} Loss: {}".format(cur_step, loss))

        epoch_avg_loss = round(epoch_sum_loss / len(epoch_train_data) ,6)

        print("Total avg loss: {} ".format(epoch_avg_loss))


        # top_k_metric, top_ents = val_via_embed(model, val_encode_data, val_encode_batch_data, val_decode_data,500)
        # # sim_mat = get_model_embed(model, val_encode_data, val_encode_batch_data, val_decode_data,1000)

        # # top_k_metric, top_ents = eval_top_k(sim_mat)


        # # ratio_top_k = process_top_k(top_k_metric, None)

        # print("val_hits@: {} = {}".format(top_k, top_k_metric))
        # final_ents_pairs = top_ents


        if(epoch_num > 0) and (epoch_num % 4 == 0):
        # if (epoch_num == 0):
            

            top_k_metric, top_ents = val_via_embed(model, val_encode_data, val_encode_batch_data, val_decode_data, 500)

            print("val_hits@: {} = {}".format(top_k, top_k_metric))

            final_ents_pairs = top_ents

            if (top_k_metric[0] <= max_pre):
                print("Max validation hit@1 - now: {} / max: {}".format(top_k_metric[0], max_pre))
                break

            else:
                max_pre = top_k_metric[0]

    return final_ents_pairs

def get_threshold_via_val(top_ents):

    top_ents_list = sorted(top_ents, key=lambda x:x[2], reverse=True) 


    cpu_num = multiprocessing.cpu_count()

    pool = multiprocessing.Pool(processes = cpu_num)
    results = []
    for i in range(len(top_ents_list)):
        results.append(pool.apply_async(cal_best_metric, (i + 1, top_ents_list)))

    pool.close()
    pool.join()
    res_list = []
    #print(results)
    for res in results:
        sub_f1, sub_threshold = res.get()
        res_list.append((sub_f1, sub_threshold))
    sort_res_list = sorted(res_list, key=lambda x:x[0], reverse=True) 

    assert len(sort_res_list) == len(top_ents_list)
    # print(sort_res_list[:20])
 
    print("Validation best_f1: {} threshold: {}".format(sort_res_list[0][0], sort_res_list[0][1]))
    return sort_res_list[0][1]
    #exit(0)
    # return sort_res_list[0][1]



def cal_best_metric(index, top_ents_list):
    all_num = len(top_ents_list)
    hit_num = 0
    #print(top_ents_list[:50])
    for i in range(index):
        if(top_ents_list[i][0] == top_ents_list[i][1]):
            hit_num += 1
    precision = round(hit_num / index, 3)
    recall = round(hit_num / all_num, 3)
    #print("index: {},hit_num: {}".format(index, hit_num))
    if (precision + recall > 0):
        f1 = round(2 * precision * recall / (precision + recall), 6)
        #print("index: {},hit_num: {}, pre: {} re: {} f1: {}".format(index, hit_num, precision, recall, f1))
    else:
        f1 = -1
    
    return f1, top_ents_list[index - 1][2]

# def get_threshold_via_val(top_ents):

#     top_ents_list = sorted(top_ents, key=lambda x:x[2]) 


#     cpu_num = multiprocessing.cpu_count()

#     pool = multiprocessing.Pool(processes = cpu_num)
#     results = []
#     for i in range(len(top_ents_list)):
#         results.append(pool.apply_async(cal_best_metric, (i, top_ents_list)))

#     pool.close()
#     pool.join()
#     res_list = []
#     #print(results)
#     for res in results:
#         sub_precision, sub_threshold = res.get()
#         res_list.append((sub_precision, sub_threshold))
#     sort_res_list = sorted(res_list, key=lambda x:x[0]) 

#     assert len(sort_res_list) == len(top_ents_list)

 
#     for res_list in sort_res_list:

#         if res_list[0] >=0.85:
#             print("Validation best_pre: {} threshold: {}".format(res_list[0], res_list[1]))
#             return res_list[1]


#     print("No threshold can make val pre >= 0.85, the best pre:{} threshold: {}".format(sort_res_list[-1][0], sort_res_list[-1][1]))
#     return sort_res_list[-1][1]
#     #exit(0)
#     # return sort_res_list[0][1]


# def cal_best_metric(index, top_ents_list):
#     all_num = len(top_ents_list)
#     hit_num = 0
#     #print(top_ents_list[:50])
#     for i in range(index, all_num):
#         if(top_ents_list[i][0] == top_ents_list[i][1]):
#             hit_num += 1
#     precision = round(hit_num / (all_num - index), 3)

#     # recall = round(hit_num / all_num, 3)
#     # #print("index: {},hit_num: {}".format(index, hit_num))
#     # if (precision + recall > 0):
#     #     f1 = round(2 * precision * recall / (precision + recall), 6)
#     #     #print("index: {},hit_num: {}, pre: {} re: {} f1: {}".format(index, hit_num, precision, recall, f1))
#     # else:
#     #     f1 = -1
    
#     return precision, top_ents_list[index][2]

            
