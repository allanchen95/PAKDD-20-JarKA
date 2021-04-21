from pyjarowinkler import distance
import pickle
import numpy as np
from data_process import *

class mt_batchs():
    def __init__(self):
        self.inputs = []
        self.len = []
        self.ent_len = []
        self.att = []
        # self.att_value = []


def pro_mt_pad(true_set, ent_att, word2idx, key2id):

    pad_ent_len = 20
    max_att_words = 20

    ent_att_matrix = []
    total_length = []
    ent_length = []
    key_list = []
    att_value_list = []
    unk_int = word2idx['<UNK>']
    pad_int = word2idx['<PAD>']

    for i in range(len(true_set)):
        ent = str(true_set[i])
        att_list =  ent_att[ent]
        # per_ent_mat = np.zeros([pad_ent_len, max_att_words], dtype=np.int32)
        ent_length.append(len(att_list))
        tmp_key = []
        tmp_value = []
        for key, value_list in att_list.items():
            if (key2id.get(key) == None):
                print("No match attr!")
                exit(0)
            
            tmp_key.append(key)

            att = ''
            for value in value_list:
                att += value + ' '
            att = att.strip().split()[:20]
            att_value = ' '.join(att)
            # print(key + "  ---  " + att_value)
            tmp_value.append(att_value)
            att2id = list(map(lambda words : word2idx.get(words, unk_int), att))
            total_length.append(len(att2id))
            att2id = att2id + [pad_int] * (max_att_words - len(att2id))
            # print(att)
            ent_att_matrix.append(att2id)
        assert len(tmp_key) == len(att_list) == len(tmp_value)
        key_list.append(tmp_key)
        att_value_list.append(tmp_value)

    # print(key_list)
    # print('--------')
    # print(att_value_list)
    # exit()

    return ent_att_matrix, total_length, ent_length, key_list, att_value_list

def get_nmt_embed(model, encode_data, encode_batch_data, decode_data, batch_size):
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
                    
    del decode_embed
    del encode_embed
    del encode_len

    gc.collect()

    return encode_pad_mat, decode_pad_mat



def pad_for_nmt(att_model, ent2att_dict1, ent2att_dict2, ent_pairs, word2idx, key2id):
    
    encode_data = mt_batchs()
    decode_data = mt_batchs()
    zh_list = []
    en_list = []

    for pair in ent_pairs:
        zh = str(pair[0])
        en = str(pair[1])
        if zh in ent2att_dict1 and en in ent2att_dict2:
            zh_list.append(zh)
            en_list.append(en)

    assert len(zh_list) == len(en_list)



    zh_ent_att_matrix, zh_total_length, zh_ent_length, encode_key_list, encode_value_list = pro_mt_pad(zh_list, ent2att_dict1, word2idx, key2id)
    en_ent_att_matrix, en_total_length, en_ent_length, decode_key_list, decode_value_list = pro_mt_pad(en_list, ent2att_dict2, word2idx, key2id)

    encode_data.inputs = zh_ent_att_matrix
    encode_data.len = zh_total_length
    encode_data.ent_len = zh_ent_length
    encode_data.att = encode_key_list
    # encode_data.att_value = encode_value_list

    decode_data.inputs = en_ent_att_matrix
    decode_data.len = en_total_length
    decode_data.ent_len = en_ent_length
    decode_data.att = decode_key_list
    # decode_data.att_value = decode_value_list

    # print(encode_data.inputs)
    # print(encode_data.len)
    # print(encode_data.ent_len)
    # print(encode_data.att)
    # print('----------')
    # print(encode_data.att_value)
    # print(decode_data.att_value)
    # exit()

    summ = 0
    for att_list in encode_data.att:
        summ += len(att_list)

    assert len(encode_data.inputs) == len(encode_data.len) == np.sum(np.array(encode_data.ent_len)) == summ

    encode_batch_data = createbatchs_for_embed(encode_data, 1000)
    
    # print(len(encode_batch_data))

    encode_pad_mat, decode_pad_mat = get_nmt_embed(att_model, encode_data, encode_batch_data, decode_data, 500)

    return encode_pad_mat, decode_pad_mat, encode_key_list, decode_key_list, encode_value_list, decode_value_list

def get_align_att_pair(att_model, key2id, ent2att_dict1, ent2att_dict2, ent_pairs, set_att_align_before, word2idx):

    encode_pad_mat, decode_pad_mat, encode_key_list, decode_key_list, encode_value_list, decode_value_list = pad_for_nmt(att_model, ent2att_dict1, ent2att_dict2, ent_pairs, word2idx, key2id)
    dict_att_align = dict()
    set_att_align = set()
    assert len(encode_pad_mat) == len(decode_pad_mat) == len(encode_key_list) == len(decode_key_list) == len(encode_value_list) == len(decode_value_list)
    # print(encode_key_list)
    # print(decode_key_list)
    # val_f = open('match_val.txt','a')
    count_value = 0
    for index in range(len(encode_pad_mat)):
        tmp_encode_embed = encode_pad_mat[index]
        tmp_decode_embed = decode_pad_mat[index]
        assert len(tmp_encode_embed) == len(tmp_decode_embed)
        len_len = len(tmp_encode_embed)
        for i in range(len(encode_key_list[index])):
            for j in range(len(decode_key_list[index])):

                score = np.dot(tmp_encode_embed[i], tmp_decode_embed[j])
                # print("{} --- {} score: {}".format(encode_key_list[index][i], decode_key_list[index][j], score))
                if(score > 0.9):
                    count_value += 1
                    att1 = encode_key_list[index][i]
                    att2 = decode_key_list[index][j]

                    val1 = encode_value_list[index][i]
                    val2 = decode_value_list[index][j]
                    tmp = '( ' + att1 +' ) ' + val1 + ' --- ' + val2 + ' ( ' + att2 +' )' + '\n'
                    # val_f.write(tmp)
                    # print("att1: {} att2: {} score: {}".format(att1, att2, score))
                    if dict_att_align.get((att1, att2)) == None:
                        dict_att_align[(att1,att2)] = 1
                    else:
                        dict_att_align[(att1,att2)] += 1
    print("count_value: ", count_value)
    # val_f.close()
    print("len(dict_att_align.keys())",len(dict_att_align.keys()))
    list_res=[]
    for pairs in dict_att_align:
        list_res.append((pairs,dict_att_align[pairs]))
    list_res.sort(key=lambda x:x[1],reverse=True)
    list_res_len = len(list_res)
    if(list_res_len == 0):
        return set_att_align
    threshold_index = int(list_res_len * 0.1)
    threshold_frequency = list_res[threshold_index][1] #len(ent_pairs)*0.04
    print("this time threshold frequency is ",threshold_frequency)
    for list_ins in range(threshold_index + 1):
        att1, att2 = list_res[list_ins][0]
        frequency = list_res[list_ins][1]
        if(key2id[att1]['id'] != key2id[att2]['id']):
            set_att_align.add((att1,att2, frequency))
            # print("Aligned att: {} / {} frequency: {}".format(att1,att2,frequency))            
    print("current align att num before one-one mapping",len(set_att_align))

    set_att_align = merge_align(set_att_align)

    set_att_align |= set_att_align_before
    print("total att-aligned num", len(set_att_align))
    
    return set_att_align

def merge_align(align):
    dict1 = {}
    dict1_set = set()
    
    dict2 = {}
    dict2_set = set()

    for att1, att2, fre in align:

        att_j = dict1.get(att1, set())
        att_j.add((att2,fre))
        dict1[att1] = att_j

        # att_i = dict2.get(att2, set())
        # att_i.add((att1, fre))
        # dict2[att2] = att_i

    ####
    for i, att_j in dict1.items():
        if(len(att_j) == 1):
            for att_fre in att_j:
                att = att_fre[0]
                fre = att_fre[1]
                dict1_set.add((i, att, fre))
        else:
            max_i = -1
            max_fre = -1 
            for att_fre in att_j:
                att = att_fre[0]
                fre = att_fre[1]
                if(fre > max_fre):
                    max_fre = fre
                    max_i = att_fre
            dict1_set.add((i, max_i[0], max_i[1]))
            # new_dict[max_i[0]] = (j, max_i[1])
    print("current align att num after left one-one mapping", len(dict1_set))

    for att1, att2, fre in dict1_set:

        att_i = dict2.get(att2, set())
        att_i.add((att1, fre))
        dict2[att2] = att_i

    for j, att_i in dict2.items():
        if(len(att_i) == 1):
            for att_fre in att_i:
                att = att_fre[0]
                fre = att_fre[1]
                dict2_set.add((att, j))
        else:
            max_i = -1
            max_fre = -1 
            for att_fre in att_i:
                att = att_fre[0]
                fre = att_fre[1]
                if(fre > max_fre):
                    max_fre = fre
                    max_i = att_fre
            dict2_set.add((max_i[0], j))
            # new_dict[max_i[0]] = (j, max_i[1])
    print("current align att num after right one-one mapping", len(dict2_set))
    print("-------final align-att----------")
    for att1, att2 in dict2_set:
        print("{} = {}".format(att1, att2))

    return dict2_set



def change_att2id_with_set_att_align(key2id, set_att_align):
    dict_key2id = key2id
    list_dict_key2id_keys = list(dict_key2id.keys())
    for att1, att2 in set_att_align:
        index1 = dict_key2id[att1]['id']
        index2 = dict_key2id[att2]['id']
        if index1 != index2:
            index = min(index1,index2)
            for one in list_dict_key2id_keys:
                if dict_key2id[one]['id'] == index1 or dict_key2id[one]['id'] == index2:
                    dict_key2id[one]['id'] = index
    pickle.dump(dict_key2id, open("temp_key2id.p","wb"))
    return dict_key2id

# def get_align_att_pair(att_model, key2id, ent2att_dict1, ent2att_dict2, ent_pairs, set_att_align_before):
#     dict_att_align = dict()
#     num = 0
#     for ent_int1, ent_int2 in ent_pairs:
#         ent_1 = str(ent_int1); ent_2 = str(ent_int2)
#         if ent_1 in ent2att_dict1 and ent_2 in ent2att_dict2:
#             for att1 in ent2att_dict1[ent_1]:
#                 for att2 in ent2att_dict2[ent_2]:
#                     flag=False
#                     for val1 in ent2att_dict1[ent_1][att1]:
#                         for val2 in ent2att_dict2[ent_2][att2]:
#                             if distance.get_jaro_distance(val1.replace(' ',''),val2.replace(' ',''),winkler=False,scaling=0.1)>0.95:
#                                 #print(att1,att2,val1,val2)
#                                 flag=True
#                     if flag==True:
#                         if (att1,att2) not in dict_att_align:
#                             dict_att_align[(att1,att2)]=0
#                         dict_att_align[(att1,att2)]+=1
#         else:
#             num+=1
#     print(num)
#     print("len(dict_att_align.keys())",len(dict_att_align.keys()))
#     list_res=[]
#     for pairs in dict_att_align:
#         list_res.append((pairs,dict_att_align[pairs]))
#     list_res.sort(key=lambda x:x[1],reverse=True)
#     threshold_frequency=200#len(ent_pairs)*0.04
#     print("this time threshold frequency is ",threshold_frequency)
#     set_att_align=set()
#     for pairs,fre in list_res:
#         att1,att2=pairs
#         if att1!=att2 and fre>threshold_frequency and ((att1,att2) not in set_att_align_before):
#             set_att_align.add((att1,att2))
#             print("++align++att",att1,att2,fre)
#             if(key2id[att1]['id'] != key2id[att2]['id']):
#                 print("NOT ALIGNED ++align++att",att1,att2,fre)
            
#     print("align att num",len(set_att_align))
#     set_att_align|=set_att_align_before
#     return set_att_align

# def change_att2id_with_set_att_align(key2id,set_att_align):
#     dict_key2id=key2id
#     list_dict_key2id_keys=list(dict_key2id.keys())
#     for att1,att2 in set_att_align:
#         index1=dict_key2id[att1]['id']
#         index2=dict_key2id[att2]['id']
#         if index1!=index2:
#             index=min(index1,index2)
#             for one in list_dict_key2id_keys:
#                 if dict_key2id[one]['id']==index1 or dict_key2id[one]['id']==index2:
#                     dict_key2id[one]['id']=index
#     pickle.dump(dict_key2id,open("temp_key2id.p","wb"))
#     return dict_key2id

