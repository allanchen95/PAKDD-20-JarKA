from __future__ import division
import numpy as np
from collections import defaultdict
import math
import multiprocessing
import time
import gc
import random
import copy
import json
import pickle

np.set_printoptions(suppress=True)

class triples:
    def __init__(self,triples,ori_triples = None):
        self.triples = triples
        self.triple_list = list(triples)
        self.triples_num = len(triples)

        self.h = set(triples[0] for triples in self.triple_list)
        self.r = set(triples[1] for triples in self.triple_list)
        self.t = set(triples[2] for triples in self.triple_list)

        self.total_entities = self.h | self.t

        self.r_list = list(self.r)
        self.ent_list = list(self.total_entities)

        if ori_triples is None:
            self.ori_triples = None
        else:
            self.ori_triples = ori_triples

        self._generate_related_ent()


    def _generate_related_ent(self):
        self.h_related_dict = defaultdict(set)
        self.t_realted_dict = defaultdict(set)

        self.hr_related_dict = defaultdict(set)
        self.rt_related_dict = defaultdict(set)

        self.ht = set()

        self.r_related_dict = defaultdict(set)

        for h,r,t in self.triple_list:
            self.h_related_dict[h].add(t)
            self.t_realted_dict[t].add(h)
            
            self.rt_related_dict[h].add((r,t))
            self.hr_related_dict[t].add((h,r))

            self.ht.add((h,t))
            self.r_related_dict[r].add((h,t))

def get_counter_dict(test_ent_pairs):
    c_dict = {}

    for pairs in test_ent_pairs:
        c_dict[pairs[0]] = pairs[1]

    return c_dict

def get_test_ents_index(ent_pair):
    # en_ent_att = pickle.load(open(folder + 'filter_sort_en_att.p', "rb"))

    # zh_ent_att = pickle.load(open(folder + 'filter_sort_zh_att.p', "rb"))

    count = 0
    zh_true_set = []
    en_true_set = []
    for pairs in ent_pair:
        zh_ent = pairs[0]
        en_ent = pairs[1]

        zh_true_set.append((zh_ent))
        en_true_set.append((en_ent))

    print("vaild/test set:", len(zh_true_set))
    # print("zh_true_set: ",zh_true_set)
    # print("en_true_set: ",en_true_set)
    # exit(0)
    # encode_data, decode_data = get_padding_data(zh_true_set, en_true_set, en_ent_att, zh_ent_att, word2idx)

    # return encode_data, decode_data, zh_true_set, en_true_set

    return zh_true_set, en_true_set

def process_data(folder):
    ori_triples1, ori_triples2, seed_sup_ent, val_ent, test_ent, _, ent_n, rel_n = read_dataset(folder)

    new_triples1, new_triples2 = add_sup_triples(ori_triples1,ori_triples2,seed_sup_ent)


    # triples1, triples2 = ut.add_sup_triples(ori_triples1, ori_triples2, seed_sup_ent1, seed_sup_ent2)

    # model = KGE_Model(ent_n, rel_n, seed_sup_ent1, seed_sup_ent2, ref_ent1, ref_ent2, ori_triples1.ent_list, ori_triples2.ent_list, len(seed_sup_ent1) + len(ref_ent1), P.embed_size, P.learning_rate)
    return ori_triples1, ori_triples2, new_triples1, new_triples2, seed_sup_ent, val_ent, test_ent, ent_n,rel_n

def find_new_triples(sup_pair1, sup_pair2, triples):
    new_triples = set()

    rt_set = triples.rt_related_dict[sup_pair1]
    if(len(rt_set) > 0):
        for r,t in rt_set:
            new_triples.add((sup_pair2,r,t))

    hr_set = triples.hr_related_dict[sup_pair1]
    if(len(hr_set) > 0):
        for h,r in hr_set:
            new_triples.add((h,r,sup_pair2))

    return new_triples

def find_new_triples_via_rel(rel1, rel2, triples):
    new_triples = set()

    ht_set = triples.r_related_dict[rel1]
    if(len(ht_set) > 0):
        for h,t in ht_set:
            new_triples.add((h,rel2,t))

    return new_triples

def generate_sup_triples(triples1, triples2, sup_pairs):



    sup_triples1, sup_triples2 = set(), set()
    sup_pairs_list = list(sup_pairs)
    for i in range(len(sup_pairs_list)):
        sup_triples1 = sup_triples1 | find_new_triples(sup_pairs_list[i][0], sup_pairs_list[i][1], triples1)
        sup_triples2 = sup_triples2 | find_new_triples(sup_pairs_list[i][1], sup_pairs_list[i][0], triples2)

    print("sup add triples: {}, {}".format(len(sup_triples1), len(sup_triples2)))

    return sup_triples1, sup_triples2

def add_sup_triples(ori_triples1, ori_triples2, sup_pairs):
    sup_triples1, sup_triples2 = generate_sup_triples(ori_triples1, ori_triples2, sup_pairs)

    new_triples1 = triples(ori_triples1.triples | sup_triples1, ori_triples = ori_triples1.triples)
    new_triples2 = triples(ori_triples2.triples | sup_triples2, ori_triples = ori_triples2.triples)

    print("ori + sup triples: {}, {}".format(len(new_triples1.triples), len(new_triples2.triples)))
    return new_triples1, new_triples2

def get_add_newly_triples(aligned_pairs, aligned_rel_pairs, model, triples1, triples2, test_index1, test_index2):
    # test_ents_list = list(model.test_ents)
    # test_ents_array = np.array(test_ents_list)
    # test_ents1 = test_ents_array[:,0]
    # test_ents2 = test_ents_array[:,1]

    # new_ents1 = []
    # new_ents2 = []
    # for key, value in aligned_pairs.items():
    #     new_ents1.append(key)
    #     new_ents2.append(value)

    # assert len(new_ents1) == len(new_ents2)
    
    newly_triples1 = set()
    newly_triples2 = set()
    # print(len(aligned_pairs))

    for ent1_index, ent2_index in aligned_pairs.items():
        # newly_triples1 = newly_triples1 | find_new_triples(test_ents1[ent1_index], test_ents2[ent2_index], triples1)
        # newly_triples2 = newly_triples2 | find_new_triples(test_ents2[ent2_index], test_ents1[ent1_index], triples2)
        #print("{}/{}".format(ent1_index, ent2_index))
        newly_triples1 = newly_triples1 | find_new_triples(ent1_index, ent2_index, triples1)
        newly_triples2 = newly_triples2 | find_new_triples(ent2_index, ent1_index, triples2)

    newly_triples1_class = triples(triples1.triples | newly_triples1)
    newly_triples2_class = triples(triples2.triples | newly_triples2)

    print("ori_triples num: 1: {} / 2: {}".format(len(triples1.triples), len(triples2.triples)))
    print("after entity switch triples: 1: {} / 2: {}".format(len(newly_triples1_class.triples), len(newly_triples2_class.triples)))

    newly_triples1 = set()
    newly_triples2 = set()

    for rel1_index, rel2_index in aligned_rel_pairs.items():
        # newly_triples1 = newly_triples1 | find_new_triples(test_ents1[ent1_index], test_ents2[ent2_index], triples1)
        # newly_triples2 = newly_triples2 | find_new_triples(test_ents2[ent2_index], test_ents1[ent1_index], triples2)
        #print("{}/{}".format(ent1_index, ent2_index))
        newly_triples1 = newly_triples1 | find_new_triples_via_rel(rel1_index, rel2_index, newly_triples1_class)
        newly_triples2 = newly_triples2 | find_new_triples_via_rel(rel2_index, rel1_index, newly_triples2_class)

    # print("ori_triples num: 1: {} / 2: {}".format(len(triples1.triples), len(triples2.triples)))
    newly_aligened_triples1 = triples(newly_triples1 | newly_triples1_class.triples)
    newly_aligened_triples2 = triples(newly_triples2 | newly_triples2_class.triples)

    print("after relation switch triples: 1: {} / 2: {}".format(len(newly_aligened_triples1.triples), len(newly_aligened_triples2.triples)))

    # newly_aligened_triples1 = triples(newly_triples1)
    # newly_aligened_triples2 = triples(newly_triples2)

    return newly_aligened_triples1, newly_aligened_triples2




def read_dataset(folder):

    triples_set1 = read_triples(folder + 'triples_1')
    triples_set2 = read_triples(folder + 'triples_2')

    triples1 = triples(triples_set1)
    triples2 = triples(triples_set2)

    total_ent_num = len(triples1.total_entities | triples2.total_entities)

    total_rel_num = len(triples1.r | triples2.r)

    total_triples_num = len(triples1.triple_list) + len(triples2.triple_list)

    print("total_ent: ",total_ent_num)
    print("total_rel_num: ",total_rel_num)
    print("total_triples: {} + {} = {}".format(len(triples1.triple_list), len(triples2.triple_list), total_triples_num))

    ref_ent_pairs = read_ref(folder + 'ref_pairs')
    print("align pairs: ",len(ref_ent_pairs))

    # test_ent_pairs = read_ref(folder + 'ref_pairs_for_mtranse_zh')
    # print("test align pairs: ",len(test_ent_pairs))

    ref_len = len(ref_ent_pairs)

    ref_ent_pairs_list = list(ref_ent_pairs)

    val_index = random.sample(range(ref_len), 1000)

    # # val_index = range(1000)


    val_ent_pairs_list = [ref_ent_pairs_list[i] for i in val_index]

    print("val_pairs: ",len(val_ent_pairs_list))

    val_ent_pairs = set(val_ent_pairs_list)

    # with open('dbp_15k_1_zh_val.pkl', 'wb') as files:
    #     pickle.dump(val_ent_pairs, files)


    # test_index = range(1000,2000)

    # test_ent_pairs_list = [ref_ent_pairs_list[i] for i in test_index]

    # test_ent_pairs = set(test_ent_pairs_list)

    # with open(folder + 'dbp_15k_1_zh_val.pkl', 'rb') as files:
    #     val_ent_pairs = pickle.load(files)
    # print("val_pairs: ",len(val_ent_pairs))
    test_ent_pairs = ref_ent_pairs - val_ent_pairs

    print("test_pairs: ",len(test_ent_pairs))

    sup_ent_pairs = read_ref(folder + 'sup_pairs')
    # print("sup_pairs: ",len(sup_ent_pairs))
    # sup_ent_pairs_list = list(sup_ent_pairs)[:2]
    # sup_ent_pairs = set(sup_ent_pairs_list)
    print("cut sup_pairs: ",len(sup_ent_pairs)) 


    return triples1, triples2, sup_ent_pairs, val_ent_pairs, test_ent_pairs, total_triples_num,total_ent_num,total_rel_num 


def read_triples(file_name):
    triples = set()
    with open(file_name, 'r') as file:
         for line in file:
             token = line.strip('\n').split('\t')
             assert len(token) == 3
             h = int(token[0])
             r = int(token[1])
             t = int(token[2])

             triples.add((h,r,t))

    return triples

def read_ref(file_name):
    ref_pairs = set()
    with open(file_name,'r') as file:
        for line in file:
            token = line.strip('\n').split('\t')
            assert len(token) == 2
            ref_pairs.add((int(token[0]), int(token[1])))

    return ref_pairs


def generate_related_matrix(triples1, triples2, ref_ent):
    ref_ent = np.array(list(ref_ent))
    ref_ent1 = ref_ent[:,0]
    ref_ent2 = ref_ent[:,1]

    print("ref_ent1: {} ref_ent2: {}".format(len(ref_ent1),len(ref_ent2)))

    h_related_matrix = np.zeros([len(ref_ent1), len(ref_ent2)])
    for i in range(len(ref_ent1)):
        ent1 = ref_ent1[i]
        related_ent1 = triples1.h_related_dict[ent1]
        for j in range(len(ref_ent2)):
            ent2 = ref_ent2[i]
            related_ent2 = triples2.h_related_dict[ent2]

            common_ents = related_ent1 & related_ent2
            if(len(common_ents)>0) and i!=j:
                h_related_matrix[i][j] = len(common_ents)

    print("related_ref_ents:", len(np.where(h_related_matrix > 0)[0]))


def find_potential_alignment_wo_cont(model, threshold):

    s_t = time.time()
    # true_aligned_set = set()
    test_sim_matrix = model.eval_test_sim_matrix()
    print("test_sim_matrix_shape:{}".format(test_sim_matrix.shape))
    tmp_dict = {}
    # tmp_dict1 = {}
    top_ents = set()
    # aligned_test_pairs = find_alignment_via_sim(test_sim_matrix, threshold, aligned_test_pairs)
    # aligned_test_pairs = revise_alignment(test_sim_matrix, aligned_test_pairs)

    for i in range(len(test_sim_matrix)):
        rank = np.argsort(-test_sim_matrix[i,:])

        score = test_sim_matrix[i][rank[0]]
        index1_id = model.test_index1[i]
        index2_id = model.test_index2[rank[0]]    
        top_ents.add((index1_id, index2_id, score))


    tmp_dict = find_alignment_via_sim(model.c_dict, top_ents, threshold)

    # tmp_dict = revise_alignment(model.c_dict, top_ents, tmp_dict)

    del test_sim_matrix
    del top_ents
    # del tmp_dict1
    # del tmp_dict
    gc.collect()
    return tmp_dict



def find_potential_alignment(model, threshold):

    s_t = time.time()
    # true_aligned_set = set()
    test_sim_matrix = model.eval_test_sim_matrix()
    print("test_sim_matrix_shape:{}".format(test_sim_matrix.shape))
    tmp_dict = {}
    # tmp_dict1 = {}
    top_ents = set()
    # aligned_test_pairs = find_alignment_via_sim(test_sim_matrix, threshold, aligned_test_pairs)
    # aligned_test_pairs = revise_alignment(test_sim_matrix, aligned_test_pairs)

    for i in range(len(test_sim_matrix)):
        rank = np.argsort(-test_sim_matrix[i,:])

        score = test_sim_matrix[i][rank[0]]
        index1_id = model.test_index1[i]
        index2_id = model.test_index2[rank[0]]    
        top_ents.add((index1_id, index2_id, score))


    tmp_dict = find_alignment_via_sim(model.c_dict, top_ents, threshold)

    tmp_dict = revise_alignment(model.c_dict, top_ents, tmp_dict)


    # num = 0

    # for key, value in tmp_dict.items():
    #     tmp_dict1[key] = value[0]

    # assert len(tmp_dict) == len(tmp_dict1)

    # aligned_test_pairs = {**aligned_test_pairs, **tmp_dict1}

    # for x, y in aligned_test_pairs.items():
    #     if model.c_dict[x] == y:
    #         num += 1
    # print("whole aligned_pairs, right alignment: {}/{}={:.3f}".format(num, len(aligned_test_pairs), num / len(aligned_test_pairs)))

    # len_n = len(model.test_ents)
    # precision = round(num / len(aligned_test_pairs), 5)
    # recall = round(num / len_n, 5)
    # if (precision + recall > 0):
    #     f1 = round(2 * precision * recall / (precision + recall), 6)
    # else:
    #     f1 = -1
    # print("Whole precision={}, recall={}, f1={}".format(precision, recall, f1))

    # delete_aligned_ents(model, tmp_dict)

    del test_sim_matrix
    del top_ents
    # del tmp_dict1
    # del tmp_dict
    gc.collect()
    return tmp_dict

def find_potential_relations(model, relation1, relation2, aligned_rel_pairs):

    s_t = time.time()
    # true_aligned_set = set()
    rel_matrix = model.eval_rel_sim()
    print("rel_matrix_shape:{}".format(rel_matrix.shape))

    dict_rel_align = dict()

    for i in range(len(rel_matrix)):
        rank = np.argsort(-rel_matrix[i,:])
        # print(rank[:10])
        # print(rel_matrix[i][rank[:10]])
        # exit()

        score = rel_matrix[i][rank[0]]
        
        if(score > 0.95):
            value = (rank[0], score)
            dict_rel_align[i] = value


    # print("len(dict_rel_align.keys())",len(dict_rel_align.keys()))
    # list_res=[]
    # for pairs in dict_rel_align:
    #     print("rel: {} - {} fre: {}".format(pairs[0], pairs[1], dict_rel_align[pairs]))
        # list_res.append((pairs,dict_att_align[pairs]))
    reverse_aligned_rels = {}
    new_aligened_rels = {}

    for i, j in dict_rel_align.items():
        rel_j = reverse_aligned_rels.get(j[0] ,set())
        rel_j.add((i,j[1]))
        reverse_aligned_rels[j[0]] = rel_j

    for j, rel_j in reverse_aligned_rels.items():
        if(len(rel_j) == 1):
            for i in rel_j:
                key = i[0]
                value = (j, i[1])
                new_aligened_rels[key] = value 
        else:
            max_i = -1
            max_score = -100 
            for i in rel_j:
                if(i[1] > max_score):
                    max_score = i[1]
                    max_i = i

            new_aligened_rels[max_i[0]] = (j, max_i[1])

    for i, j in new_aligened_rels.items():
        rel1_index = i
        rel2_index = int(j[0] + len(relation1))
        # if(relation1[rel1_index] != relation2[rel2_index]):
        #     print("rel: {} - {} fre: {}".format(relation1[rel1_index], relation2[rel2_index], round(j[1], 3)))
        
        aligned_rel_pairs[rel1_index] = rel2_index
    print("total_aligned: ",len(aligned_rel_pairs))
    del rel_matrix
    gc.collect()
    # return tmp_dict
    return aligned_rel_pairs



def delete_aligned_ents(model, tmp_dict):


    m_test_ents1 = copy.deepcopy(model.test_index1)
    m_test_ents2 = copy.deepcopy(model.test_index2)

    for i, j in tmp_dict.items():
        # align_set.add((test_ents1[i],test_ents2[j]))
        # print("pop:{}/ {}".format(i, j))
        m_test_ents1.remove(i)
        m_test_ents2.remove(j)

    model.test_index1 = m_test_ents1
    model.test_index2 = m_test_ents2



def find_alignment_via_sim(c_dict, top_ents, threshold):


    total_num = 0
    hit_num = 0
    len_n = len(top_ents)
    top_ents_list = sorted(top_ents, key=lambda x:x[2], reverse=True)
    tmp_dict = {}

    for i in range(len(top_ents_list)):
        pairs = top_ents_list[i]
        if(pairs[2] >= threshold):
            total_num += 1
            value = (pairs[1], i)
            tmp_dict[pairs[0]] = value
        else:
            break

    # for pairs in top_ents_list:
    #     if (pairs[2] >= threshold):
    #         total_num += 1
    #         # aligned_test_pairs[pairs[0]] = pairs[1]
    #         tmp_dict[pairs[0]] = pairs[1]
    #         # if c_dict[pairs[0]] == pairs[1]:
    #         #     hit_num += 1
    #     else:
    #         break

    if(total_num > 0):
        check_alignment(c_dict, tmp_dict, "After test, before revised: ")
        return tmp_dict
    else:
        print("No more test seed proposed!")
        # exit(0)
        return {}

def revise_alignment(c_dict, top_ents, aligned_pairs):
    reverse_aligned_pairs = {}
    new_aligened_pairs = {}
    check_dict = {}
    for i,j,k in top_ents:
        key = str(i) + '/' + str(j)
        check_dict[key] = k

    # print(aligned_pairs)
    # print("-----------")
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
    # print(new_aligened_pairs)
    check_alignment(c_dict, new_aligened_pairs, "After revised: ")

    return new_aligened_pairs

# def check_alignment(aligned_pairs, all_num, context = ""):
#     if aligned_pairs is None or len(aligned_pairs) == 0:
#         print("{}, Empty aligned pairs".format(context))
#         return
#     num = 0
#     for x, y in aligned_pairs.items():
#         if x == y:
#             num += 1
#     print("{}, right alignment: {}/{}={:.3f}".format(context, num, len(aligned_pairs), num / len(aligned_pairs)))

#     precision = round(num / len(aligned_pairs), 5)
#     recall = round(num / all_num, 5)
#     f1 = round(2 * precision * recall / (precision + recall), 6)
#     print("precision={}, recall={}, f1={}".format(precision, recall, f1))


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



def train_transe(triples1, triples2, neg_num, model, n, batch_size, max_train_num):
    print("Train_triples1: {} Train_triples2: {}".format(len(triples1.triples), len(triples2.triples)))
    ori_pre = 0    
    for train_num in range(max_train_num):
        print("train epoch: ", train_num + 1)
        new_pre = train_transe_epoch_n_epo(triples1, triples2, neg_num, model, n, batch_size)
        if(new_pre - ori_pre) < 0:
            break
        else:
            ori_pre = new_pre
        # flag, hit_1, test_threshold = model.get_threshold_via_val()
        # print_threshold_func(flag, hit_1, test_threshold, model)

def train_transe_epoch_n_epo(triples1, triples2, neg_num, model, n, batch_size):

    kb_ents1_embed = model.get_kb1_embed()
    kb_ents2_embed = model.get_kb2_embed()
    # print("kb1: ", kb_ents1_embed.shape)
    # print("kb2: ", kb_ents2_embed.shape)
    s_t = time.time()
    nei_kb1 = generate_neighbors(kb_ents1_embed, model.kb_ents1, neg_num)
    nei_kb2 = generate_neighbors(kb_ents2_embed, model.kb_ents2, neg_num)
    e_t = time.time()
    print("Generate neighbors --- kb1_n: {} kb2_n: {} time: {}".format(len(nei_kb1), len(nei_kb2), round(e_t - s_t,3)))
    for i in range(n):
        loss, epoch_time = train_transe_epoch_1_epo(triples1, triples2, nei_kb1, nei_kb2, batch_size, neg_num, model)
        print("Train loss = {:.3f}, time = {:.3f} ".format(loss, epoch_time))
    print("Begin validate")
    pre = model.validate()
    return pre
    #model.test()




def print_threshold_func(flag, hit_1, test_threshold, model):

    if(flag == 1):
        print("Validation Precison: {}, Threshold: {}".format(hit_1, test_threshold))
    else:
        print("No threshold exceed 85%; max_precision: {} threshold: {}".format(hit_1, test_threshold))
        #model.test()
        exit(0)

def train_transe_via_newly_alignment(triples1, triples2, model, epoch, batch_size):
    
    print("New seed triples: {}, {}".format(len(triples1.triples), len(triples2.triples)))    

    total_triples_num = len(triples1.triples) + len(triples2.triples)

    train_step = math.ceil(total_triples_num / batch_size)

    for i in range(epoch):
        s_t = time.time()
        alignment_loss = 0
        for step in range(train_step):
            batch_pos = []
            pos_triples1, pos_triples2 = get_pos_batch(triples1.triple_list, triples2.triple_list, step, batch_size)
            batch_pos.extend(pos_triples1)
            batch_pos.extend(pos_triples2)

            feed_dict = {
                        model.new_h : [x[0] for x in batch_pos],
                        model.new_r : [x[1] for x in batch_pos],
                        model.new_t : [x[2] for x in batch_pos]
            }
            loss, _ = model.sess.run([model.alignment_loss, model.alignment_op], feed_dict = feed_dict)
            alignment_loss += loss
        alignment_loss /= train_step
        e_t = time.time()
        time_cost = round(e_t - s_t, 3)
        print("Alignment loss = {:.3f}, time = {:.3f} ".format(alignment_loss, time_cost))




def train_transe_epoch_1_epo(triples1, triples2, nei_kb1, nei_kb2, batch_size, neg_num, model):
    s_t = time.time()
    epoch_loss = 0
    triples_num = triples1.triples_num + triples2.triples_num
    #print("train triples_num: ", triples_num)

    train_step = math.ceil(triples_num / batch_size)
    #print("batch_size: {} train_step: {}".format(batch_size, train_step))

    random.shuffle(triples1.triple_list)
    random.shuffle(triples2.triple_list)

    for step in range(train_step):
        batch_pos, batch_neg = get_batch_via_neighbors(triples1, triples2, step, batch_size, nei_kb1, nei_kb2, neg_num)
        #print("step: {} batch_pos: {} batch_neg: {}".format(step, len(batch_pos), len(batch_neg)))


        feed_dict ={model.pos_hs : [x[0] for x in batch_pos],
                    model.pos_rs : [x[1] for x in batch_pos],
                    model.pos_ts : [x[2] for x in batch_pos],
                    model.neg_hs : [x[0] for x in batch_neg],
                    model.neg_rs : [x[1] for x in batch_neg],
                    model.neg_ts : [x[2] for x in batch_neg]}


        loss,_ = model.sess.run([model.triple_loss, model.triple_op], feed_dict = feed_dict)
        epoch_loss += loss
    epoch_loss /= train_step
    e_t =time.time()
    return epoch_loss, round(e_t - s_t, 3)



def get_batch_via_neighbors(triples1, triples2, step, batch_size, nei_kb1, nei_kb2, neg_num):
    pos_triples1, pos_triples2 = get_pos_batch(triples1.triple_list, triples2.triple_list, step, batch_size)
    neg_triples = []
    
    # neg_triples.extend(neg_sampling(pos_triples1, triples1.triples, nei_kb1, nei_kb2, neg_num))
    # neg_triples.extend(neg_sampling(pos_triples2, triples2.triples, nei_kb1, nei_kb2, neg_num))

    neg_triples.extend(neg_sampling(pos_triples1, triples1.triples, nei_kb1, triples1.ent_list, neg_num))
    neg_triples.extend(neg_sampling(pos_triples2, triples2.triples, nei_kb2, triples2.ent_list, neg_num))

    total_pos = []
    total_pos.extend(pos_triples1)
    total_pos.extend(pos_triples2)

    return total_pos, neg_triples

def neg_sampling(pos_triples, triples, nei_kb, ent_list, neg_num):
    sample_num = 10
    neg_triples = []
    ent_list = np.array(ent_list)
    count1 = 0
    count2 = 0
    for (h,r,t) in pos_triples:
        choice = random.randint(0,1)

        if choice == 1:
            count1 += 1
            candi = nei_kb.get(h, ent_list)
            index = random.sample(range(len(candi)), sample_num)

            neg_list = candi[index]

            for h_r in neg_list:
                if(h_r, r, t) not in triples:
                    neg_triples.append((h_r, r, t))
        else:
            count2 += 1
            candi = nei_kb.get(t, ent_list)
            index = random.sample(range(len(candi)), sample_num)
            neg_list = candi[index]

            for t_r in neg_list:
                if(h, r, t_r) not in triples:
                    neg_triples.append((h, r, t_r))
    
    # print("count1: ",count1)
    # print("count2: ",count2)
    return neg_triples





'''
def neg_sampling(pos_triples, triples, nei_kb1, nei_kb2, neg_num):
    sample_num = 10
    neg_triples = set()
    #print("pos_triples: ",len(pos_triples))
    count1 = 0
    count2 = 0
    summ = 0
    for (h,r,t) in pos_triples:
        choice = random.randint(0,1)
        # summ += 1
        # print("num: {} neg_num:{}".format(summ, len(neg_triples)))

        if choice == 1:

            if h in nei_kb1:
                count1 += 1
                index = nei_kb1[h]
                assert(len(index) == neg_num)
                sample_index = random.sample(range(len(index)), sample_num)
                neg_list = index[sample_index]
                for h_r in neg_list:
                    if((h_r, r ,t) not in triples) and ((h_r, r ,t) not in neg_triples):
                        neg_triples.add((h_r, r, t))


            else:
                count2 += 1
                index = nei_kb2[h]
                assert(len(index) == neg_num)
                sample_index = random.sample(range(len(index)), sample_num)
                neg_list = index[sample_index]        
                for h_r in neg_list:
                    if((h_r, r ,t) not in triples) and ((h_r, r ,t) not in neg_triples):
                        neg_triples.add((h_r, r, t))


        else:
            if t in nei_kb1:
                count1 += 1
                index = nei_kb1[t]
                assert(len(index) == neg_num)
                sample_index = random.sample(range(len(index)), sample_num)
                neg_list = index[sample_index]
                for t_r in neg_list:
                    if((h, r ,t_r) not in triples) and ((h, r ,t_r) not in neg_triples):
                        neg_triples.add((h, r, t_r))


            else:
                count2 += 1
                index = nei_kb2[t]
                assert(len(index) == neg_num)
                sample_index = random.sample(range(len(index)), sample_num)
                neg_list = index[sample_index]
                for t_r in neg_list:
                    if((h, r ,t_r) not in triples) and ((h, r ,t_r) not in neg_triples):
                        neg_triples.add((h, r, t_r))

    # print("count1: ",count1)
    # print("count2: ",count2)
    return list(neg_triples)        
            # index = random.sample(range(len(candi)), neg_num)
'''

def get_pos_batch(triples1_list, triples2_list, step, batch_size):

    ratio_num1 = int ((len(triples1_list) * batch_size) / (len(triples1_list) + len(triples2_list)))

    ratio_num2 = batch_size - ratio_num1
    #print("ratio_num1: {} ratio_num2: {}".format(ratio_num1, ratio_num2))

    start1 = step * ratio_num1
    start2 = step * ratio_num2

    end1 = start1 + ratio_num1
    end2 = start2 + ratio_num2

    if(end1 > len(triples1_list)):
        end1 = len(triples1_list)

    if(end2 > len(triples2_list)):
        end2 = len(triples2_list)

    pos_triples1 = triples1_list[start1:end1]
    pos_triples2 = triples2_list[start2:end2]
    return pos_triples1, pos_triples2





def generate_neighbors(embed, ent_list, neg_num):
    cpu_num = multiprocessing.cpu_count()
    split_ents = div_list(np.array(ent_list), cpu_num)
    split_ents_index = div_list(np.array(range(len(ent_list))), cpu_num)

    #print("split_num:{} total: {}".format(len(split_ents), len(split_ents[0])))

    # merge_dict = {}
    # for i in range(len(split_ents)):
    #     res = cal_sim_neighbors(split_ents[i], np.array(ent_list), embed[split_ents_index[i],:], embed, neg_num)
    #     merge_dict = {**merge_dict, **res}

    pool = multiprocessing.Pool(processes = cpu_num)
    results = []
    for i in range(len(split_ents)):
        results.append(pool.apply_async(cal_sim_neighbors, (split_ents[i], np.array(ent_list), embed[split_ents_index[i],:], embed, neg_num)))

    pool.close()
    pool.join()
    merge_dict = {}
    for res in results:
        merge_dict = {**merge_dict, **res.get()}
    del embed
    gc.collect()
    #print(len(merge_dict))
    return merge_dict    

def cal_sim_neighbors(split_ents, ents, split_ents_embed, ents_embed, neg_num):
    sim_matrix = np.matmul(split_ents_embed, ents_embed.T)
    ents_nei = {}
    for i in range(sim_matrix.shape[0]):
        sort_index = np.argpartition(-sim_matrix[i,:], neg_num - 1)
        ents_nei[split_ents[i]] = ents[sort_index[0: neg_num]]
    del sim_matrix
    gc.collect()

    return ents_nei


def div_list(ls, n):
    length = len(ls)

    batch_num = length // n
    last = length % n

    process_ls = []

    for i in range(0, (n-1)*batch_num, batch_num):
        process_ls.append(ls[i:i + batch_num])
    process_ls.append(ls[(n-1) * batch_num:])

    return process_ls



def evaluation(embed1, embed2, test_index1, test_index2, context= "", aligned_ref_pairs = None):

    s_t = time.time()
    aligned_dict = {}
    if (aligned_ref_pairs is not None) and (len(aligned_ref_pairs) > 0):
        for key, value in aligned_ref_pairs.items():
            if key not in aligned_dict:
                aligned_dict[key] = value
            else:
                print("Error!")

    top_k = [1, 3, 5, 10, 50]
    eval_metric = np.array([0 for i in top_k])
    mrr = 0

    eval_metric1 = np.array([0 for i in top_k])
    mrr1 = 0

    pre_set = set()

    ref_num = len(embed1)

    cpu_num = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes = cpu_num)

    split_ref_index = div_list(np.array(range(ref_num)), cpu_num)

    results = []
    for i in split_ref_index:
        results.append(pool.apply_async(rank_ref_via_embed,(i, embed1[i,:], embed2, top_k, aligned_dict, test_index1, test_index2)))
    pool.close()
    pool.join()

    for res in results:
        tmp_mrr, tmp_top_k, tmp_mrr1, tmp_top_k1, tmp_pre = res.get()
        mrr += tmp_mrr
        eval_metric += tmp_top_k

        mrr1 += tmp_mrr1
        eval_metric1 += tmp_top_k1

        pre_set |= tmp_pre
        # aligned_ref_pairs |= tmp_true

    assert len(pre_set) == ref_num

    ratio_top_k = np.array([0 for i in top_k], dtype = np.float32)

    for i in range(len(ratio_top_k)):
        ratio_top_k[i] = round(eval_metric[i] / ref_num, 4)

    ratio_mrr = round(mrr/ref_num, 3)

    print("{} without aligned, hits@{} = {}, mrr = {:.3f}, time = {:.3f}".format(context, top_k, ratio_top_k, ratio_mrr, time.time() - s_t))

    if (aligned_ref_pairs is not None) and (len(aligned_ref_pairs) > 0):
        ratio_top_k1 = np.array([0 for i in top_k], dtype = np.float32)

        for i in range(len(ratio_top_k1)):
            ratio_top_k1[i] = round(eval_metric1[i] / ref_num, 4)

        ratio_mrr1 = round(mrr1/ref_num, 3)

        print("{} with aligned, hits@{} = {}, mrr = {:.3f}, time = {:.3f}".format(context, top_k, ratio_top_k1, ratio_mrr1, time.time() - s_t))
    return ratio_top_k        


def rank_ref_via_embed(ref_index, sub_embed, embed, top_k, aligned_dict, test_index1, test_index2):

    sim_matrix = np.matmul(sub_embed, embed.T)

    pre_set = set()
    # true_aligned_set = set()

    mrr = 0

    mrr1 = 0

    top_k_matric = np.array([0 for k in top_k])

    top_k_matric1 = np.array([0 for k in top_k])
    
    aligned_ref = None
    for i in range(len(ref_index)):
        ref_tmp = ref_index[i]
        rank = np.argsort(-sim_matrix[i,:])

        aligned_ref = rank[0]



        true_index = np.where(rank == ref_tmp)[0][0]

        # mrr
        mrr = mrr + 1/(true_index + 1)

        # top_k:[1, 5, 10, 50]
        for k in range(len(top_k)):
            if true_index < top_k[k]:
                top_k_matric[k] += 1

        # if (len(aligned_dict) > 0) and (ref_tmp in aligned_dict):
        #     mrr1 = mrr1 + 1
        #     for k in range(len(top_k)):
                
        #         top_k_matric1[k] += 1
        if (len(aligned_dict) > 0) and aligned_dict.get(test_index1[ref_tmp]) != None:
            aligned_e = aligned_dict[test_index1[ref_tmp]]
            index = test_index2.index(aligned_e)
            sim_matrix[i][index] += 1.0
            rank = np.argsort(-sim_matrix[i,:])
            aligned_ref = rank[0]
            true_index = np.where(rank == ref_tmp)[0][0]

            # mrr
            mrr1 = mrr1 + 1/(true_index + 1)
            for k in range(len(top_k)):
                if true_index < top_k[k]:
                    top_k_matric1[k] += 1    


        else:
            mrr1 = mrr1 + 1/(true_index + 1)
            for k in range(len(top_k)):
                if true_index < top_k[k]:
                    top_k_matric1[k] += 1             



        
        pre_set.add((ref_tmp, aligned_ref))

    del sim_matrix
    gc.collect()

    return mrr, top_k_matric, mrr1, top_k_matric1, pre_set



# def cal_threshold_via_embed(val_embed1, val_embed2):
#     sim_matrix = np.matmul(val_embed1, val_embed2.T)
#     top_ents = set()

#     for i in range(len(sim_matrix)):

#         top_ent_index = np.argpartition(-sim_matrix[i,:], 0)[0]
#         score = sim_matrix[i][top_ent_index]
#         top_ents.add((i, top_ent_index, score))

#     top_ents_list = sorted(top_ents, key=lambda x:x[2])
#     # top_ents_list_print = sorted(top_ents, key=lambda x:x[2], reverse = True)
#     # # print(top_ents_list_print[:20]) 


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
#     # sort_res_list_print = sorted(res_list, key=lambda x:x[0], reverse=True) 
#     # print(sort_res_list_print[:20])

#     assert len(sort_res_list) == len(top_ents_list)

#     for res_list in sort_res_list:

#         if res_list[0] >=0.85:
#             print("Validation best_pre: {} threshold: {}".format(res_list[0], res_list[1]))
#             return res_list[1]


#     print("No threshold can make val pre >= 0.85, the best pre:{} threshold: {}".format(sort_res_list[-1][0], sort_res_list[-1][1]))
#     return sort_res_list[-1][1]

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



def cal_threshold_via_embed(val_embed1, val_embed2):
    sim_matrix = np.matmul(val_embed1, val_embed2.T)

    top_ents = set()

    for i in range(len(sim_matrix)):

        top_ent_index = np.argpartition(-sim_matrix[i,:], 0)[0]
        score = sim_matrix[i][top_ent_index]
        top_ents.add((i, top_ent_index, score))

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
 
    print("Validation best_f1: {} threshold: {}".format(sort_res_list[0][0], sort_res_list[0][1]))
    return sort_res_list[0][1]


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


def val_evaluation(embed1, embed2, context= ""):

    s_t = time.time()
    top_k = [1, 5, 10, 50]
    eval_metric = np.array([0 for i in top_k])
    mrr = 0

    pre_set = set()

    ref_num = len(embed1)

    cpu_num = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes = cpu_num)

    split_ref_index = div_list(np.array(range(ref_num)), cpu_num)

    results = []
    for i in split_ref_index:
        results.append(pool.apply_async(rank_val_via_embed,(i, embed1[i,:], embed2, top_k)))
    pool.close()
    pool.join()

    for res in results:
        tmp_mrr, tmp_top_k, tmp_pre = res.get()
        mrr += tmp_mrr
        eval_metric += tmp_top_k

        pre_set |= tmp_pre
        # aligned_ref_pairs |= tmp_true

    assert len(pre_set) == ref_num

    ratio_top_k = np.array([0 for i in top_k], dtype = np.float32)

    for i in range(len(ratio_top_k)):
        ratio_top_k[i] = round(eval_metric[i] / ref_num, 4)

    ratio_mrr = round(mrr/ref_num, 3)

    print("{} hits@{} = {}, mrr = {:.3f}, time = {:.3f}".format(context, top_k, ratio_top_k, ratio_mrr, time.time() - s_t))

    return ratio_top_k    

def rank_val_via_embed(ref_index, sub_embed, embed, top_k):

    sim_matrix = np.matmul(sub_embed, embed.T)

    pre_set = set()
    # true_aligned_set = set()

    mrr = 0


    top_k_matric = np.array([0 for k in top_k])

    
    aligned_ref = None
    for i in range(len(ref_index)):
        ref_tmp = ref_index[i]
        rank = np.argsort(-sim_matrix[i,:])

        aligned_ref = rank[0]



        true_index = np.where(rank == ref_tmp)[0][0]

        # mrr
        mrr = mrr + 1/(true_index + 1)

        # top_k:[1, 5, 10, 50]
        for k in range(len(top_k)):
            if true_index < top_k[k]:
                top_k_matric[k] += 1

        
        pre_set.add((ref_tmp, aligned_ref))

    del sim_matrix
    gc.collect()

    return mrr, top_k_matric, pre_set


def merge_seed(c_dict, str_tmp_dict, att_tmp_dict, aligned_test_pairs, strategy = "rank_based"):

    att_dict = {}
    str_dict = {}
    merge_dict = {}
    reverse_dict = {}
    new_dict = {}
    final_dict = {}
    total_keys = set()

    str_len = len(str_tmp_dict)
    att_len = len(att_tmp_dict)

    print("str: {}    att: {}".format(str_len, att_len))

    for i, j in str_tmp_dict.items():
        if strategy == "rank_based":
            ratio = round(j[1]/str_len, 6)
        elif strategy == "score_based":
            ratio = j[1]
        else:
            print("No such strategy!")
            exit()
        str_dict[i] = (j[0], ratio)
        total_keys.add((i))

    assert len(str_dict) == len(str_tmp_dict)

    for i, j in att_tmp_dict.items():
        if strategy == "rank_based":
            ratio = round(j[1]/att_len, 6)
        elif strategy == "score_based":
            ratio = j[1]
        else:
            print("No such strategy!")
            exit()
        att_dict[i] = (j[0], ratio)
        total_keys.add((i))

    assert len(att_dict) == len(att_tmp_dict)

    for key in total_keys:
        if str_dict.get(key) != None:
            if att_dict.get(key) != None:
                str_value = str_dict[key]
                att_value = att_dict[key]

                if str_value[1] > att_value[1]:
                    merge_dict[key] = att_value
                else:
                    merge_dict[key] = str_value

            else:
                merge_dict[key] = str_dict[key]
        else:
            merge_dict[key] = att_dict[key]

    num = 0
    for x, y in merge_dict.items():
        if c_dict[x] == y[0]:
            num += 1
    print("{} before revised, right alignment: {}/{}={:.3f}".format('merge_dict', num, len(merge_dict), num / len(merge_dict)))
    # print(merge_dict)

    for i, j in merge_dict.items():


        ent_j = reverse_dict.get(j[0] ,set())
        ent_j.add((i,j[1]))
        reverse_dict[j[0]] = ent_j

    for j, ent_j in reverse_dict.items():
        if(len(ent_j) == 1):
            for i in ent_j:
                key = i[0]
                value = (j, i[1])
                new_dict[key] = value 
        else:
            max_i = -1
            max_ratio = 10 
            for i in ent_j:
                if(i[1] < max_ratio):
                    max_ratio = i[1]
                    max_i = i

            new_dict[max_i[0]] = (j, max_i[1])


    num = 0
    for x, y in new_dict.items():
        if c_dict[x] == y[0]:
            num += 1
    print("{} after revised, right alignment: {}/{}={:.3f}".format('merge_dict', num, len(new_dict), num / len(new_dict)))
    # print(merge_dict)


    for i, j in new_dict.items():
        
        final_dict[i] = j[0]

    assert len(final_dict) == len(new_dict)

    num = 0
    aligned_test_pairs = {**aligned_test_pairs, **final_dict}

    for x, y in aligned_test_pairs.items():
        if c_dict[x] == y:
            num += 1
    print("whole aligned_pairs, right alignment: {}/{}={:.3f}".format(num, len(aligned_test_pairs), num / len(aligned_test_pairs)))


    len_n = len(c_dict)
    precision = round(num / len(aligned_test_pairs), 5)
    recall = round(num / len_n, 5)
    if (precision + recall > 0):
        f1 = round(2 * precision * recall / (precision + recall), 6)
        #print("index: {},hit_num: {}, pre: {} re: {} f1: {}".format(index, hit_num, precision, recall, f1))
    else:
        f1 = -1
    print("Whole precision={}, recall={}, f1={}".format(precision, recall, f1))

    return aligned_test_pairs, final_dict

