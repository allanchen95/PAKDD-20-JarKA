from __future__ import division
import numpy as np
from collections import defaultdict
import math
import tensorflow as tf
import time
import copy
from train_funcs import *

class KGE_model:
    def __init__(self, sess, ent_num, rel_num, rel_num1, rel_num2, kb_ents1, kb_ents2, val_ents, test_ents, test_index1, test_index2, c_dict, sup_ents, embed_size, lambda_1, lambda_2, mu_1, lr, max_gradient_norm = 5.0):
        self.sess = sess
        # self.aligned_ref_pairs = aligned_ref_pairs
        self.ent_num = ent_num
        self.rel_num = rel_num
        self.rel_num1 = rel_num1
        self.rel_num2 = rel_num2
        self.kb_ents1 = kb_ents1
        self.kb_ents2 = kb_ents2
        # self.ref_ents = ref_ents
        self.val_ents = val_ents
        self.test_ents = test_ents
        self.test_index1 = test_index1
        self.test_index2 = test_index2
        self.c_dict = c_dict

        self.embed_size = embed_size
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.mu_1 = mu_1
        self.lr = lr
        self.max_gradient_norm = max_gradient_norm

        self.ent_embeddings = tf.nn.l2_normalize(tf.Variable(tf.truncated_normal([self.ent_num, self.embed_size], stddev=1.0 / math.sqrt(self.embed_size))), 1)
        self.rel_embeddings = tf.nn.l2_normalize(tf.Variable(tf.truncated_normal([self.rel_num, self.embed_size], stddev=1.0 / math.sqrt(self.embed_size))), 1)

        self._build_graph()

    def _build_graph(self):
        
        def generate_loss(phs, prs, pts, nhs, nrs, nts):
            pos_score = tf.reduce_sum(tf.pow(phs + prs - pts, 2), 1)
            neg_score = tf.reduce_sum(tf.pow(nhs + nrs - nts, 2), 1)
            pos_loss = tf.reduce_sum(tf.maximum(pos_score - tf.constant(self.lambda_1), 0))
            neg_loss = self.mu_1 * tf.reduce_sum(tf.maximum(tf.constant(self.lambda_2) - neg_score, 0))

            return pos_loss, neg_loss

        # TransE loss
        self.pos_hs = tf.placeholder(tf.int32, shape=[None])
        self.pos_rs = tf.placeholder(tf.int32, shape=[None])
        self.pos_ts = tf.placeholder(tf.int32, shape=[None])
        self.neg_hs = tf.placeholder(tf.int32, shape=[None])
        self.neg_rs = tf.placeholder(tf.int32, shape=[None])
        self.neg_ts = tf.placeholder(tf.int32, shape=[None])
        phs = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_hs)
        prs = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_rs)
        pts = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_ts)
        nhs = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_hs)
        nrs = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_rs)
        nts = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_ts)

        self.pos_loss, self.neg_loss = generate_loss(phs, prs, pts, nhs, nrs, nts)
        self.triple_loss = self.pos_loss + self.neg_loss
        
        #self.triple_op = tf.train.AdagradOptimizer(self.lr).minimize(self.triple_loss)    


        optimizer = tf.train.AdamOptimizer(self.lr)
        self.trainable_params = tf.trainable_variables()
        # print("TransE")
        # print(len(trainable_params))
        # for v in trainable_params:
        #     print(v)

        gradients = tf.gradients(self.triple_loss, self.trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        self.triple_op = optimizer.apply_gradients(zip(clip_gradients, self.trainable_params))

        # Alignment loss
        self.new_h = tf.placeholder(tf.int32, shape=[None])
        self.new_r = tf.placeholder(tf.int32, shape=[None])
        self.new_t = tf.placeholder(tf.int32, shape=[None])
        new_phs = tf.nn.embedding_lookup(self.ent_embeddings, self.new_h)
        new_prs = tf.nn.embedding_lookup(self.rel_embeddings, self.new_r)
        new_pts = tf.nn.embedding_lookup(self.ent_embeddings, self.new_t)

        self.alignment_loss = -tf.reduce_sum(tf.log(tf.sigmoid(-tf.reduce_sum(tf.pow(new_phs + new_prs - new_pts, 2), 1))))

        optimizer = tf.train.AdamOptimizer(self.lr)
        self.trainable_params = tf.trainable_variables()
        # print("Alignemnt")
        # print(len(trainable_params))
        # for v in trainable_params:
        #     print(v)
        gradients = tf.gradients(self.alignment_loss, self.trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        self.alignment_op = optimizer.apply_gradients(zip(clip_gradients, self.trainable_params))

        self.test1_index = tf.placeholder(tf.int32, shape=[None])
        self.test2_index = tf.placeholder(tf.int32, shape=[None])

        self.test_ents1 = tf.nn.embedding_lookup(self.ent_embeddings, self.test1_index)
        self.test_ents2 = tf.nn.embedding_lookup(self.ent_embeddings, self.test2_index)

        self.rel1_index = tf.placeholder(tf.int32, shape=[None])
        self.rel2_index = tf.placeholder(tf.int32, shape=[None])

        self.rel1_embed = tf.nn.embedding_lookup(self.rel_embeddings, self.rel1_index)
        self.rel2_embed = tf.nn.embedding_lookup(self.rel_embeddings, self.rel2_index)    


    def eval_rel_sim(self):
        # test_ents_list = list(self.test_ents)
        # test_ents_array = np.array(test_ents_list)

        rel1_index = np.arange(self.rel_num1)
        rel2_index = np.arange(self.rel_num1, self.rel_num)

        # rel1_embed = tf.nn.embedding_lookup(self.rel_embeddings, rel1_index)
        # rel2_embed = tf.nn.embedding_lookup(self.rel_embeddings, rel2_index)

        rel_embed1, rel_embed2 = self.sess.run([self.rel1_embed, self.rel2_embed], feed_dict = {self.rel1_index: rel1_index, self.rel2_index: rel2_index})

        # print("rel1: {} rel2: {}".format(rel_embed1.shape, rel_embed2.shape))
        

        sim_matrix = np.matmul(rel_embed1, rel_embed2.T)

        return sim_matrix



    def eval_test_sim_matrix(self):
        # test_ents_list = list(self.test_ents)
        # test_ents_array = np.array(test_ents_list)
        test1_index = np.array(self.test_index1)
        test2_index = np.array(self.test_index2)

        # test_ents1 = tf.nn.embedding_lookup(self.ent_embeddings, test1_index)
        # test_ents2 = tf.nn.embedding_lookup(self.ent_embeddings, test2_index)

        test_embed1, test_embed2 = self.sess.run([self.test_ents1, self.test_ents2], feed_dict = {self.test1_index: test1_index, self.test2_index: test2_index})
        
        s_t = time.time()

        sim_matrix = np.matmul(test_embed1, test_embed2.T)
        print("test sim_matrix time: {}".format(round(time.time() - s_t ,3)))

        return sim_matrix



    def validate(self):
        # s_t = time.time()
        # val_ents_list = list(self.val_ents)
        # val_ents_array = np.array(val_ents_list)
        # val_ents1 = tf.nn.embedding_lookup(self.ent_embeddings, val_ents_array[:,0])
        # val_ents2 = tf.nn.embedding_lookup(self.ent_embeddings, val_ents_array[:,1])

        # val_embed1, val_embed2 = self.sess.run([val_ents1, val_ents2])
        
        # print("ref1: ", np.array(val_embed1).shape)
        # print("ref2: ", np.array(val_embed2).shape)

        val_ents_list = list(self.val_ents)
        val_ents_array = np.array(val_ents_list)
        
        # remain_test_ents_array = np.array(self.test_index2)

        # print(val_ents_array[:,0][:10])
        # print(val_ents_array[:,1][:10])
        # print(remain_test_ents_array[:,0][:10])
        # print(remain_test_ents_array[:,1][:10])
        val_ents2_index = list(copy.deepcopy(val_ents_array[:,1]))
        # print(len(val_ents2_index))
        val_ents2_index.extend(self.test_index2)
        val_ents2_index = np.array(val_ents2_index)
        # print(len(val_ents2_index))
        # # print(val_ents2_index[:10]) 
        # exit(0)
        val_ents1 = tf.nn.embedding_lookup(self.ent_embeddings, val_ents_array[:,0])
        # val_ents2 = tf.nn.embedding_lookup(self.ent_embeddings, val_ents_array[:,1])
        val_ents2 = tf.nn.embedding_lookup(self.ent_embeddings, val_ents2_index)

        print("val embed: {}/{}".format(val_ents1.shape, val_ents2.shape))

        val_embed1, val_embed2 = self.sess.run([val_ents1, val_ents2])

        top_k = val_evaluation(val_embed1, val_embed2, "Validation")



        del val_embed1, val_embed2
        gc.collect()

        return top_k[0]




    def get_threshold_via_val(self):

        val_ents_list = list(self.val_ents)
        val_ents_array = np.array(val_ents_list)
        
        # remain_test_ents_array = np.array(self.test_index2)

        # print(val_ents_array[:,0][:10])
        # print(val_ents_array[:,1][:10])
        # print(remain_test_ents_array[:,0][:10])
        # print(remain_test_ents_array[:,1][:10])
        val_ents2_index = list(copy.deepcopy(val_ents_array[:,1]))
        # print(len(val_ents2_index))
        val_ents2_index.extend(self.test_index2)
        val_ents2_index = np.array(val_ents2_index)
        # print(len(val_ents2_index))
        # # print(val_ents2_index[:10]) 
        # exit(0)
        val_ents1 = tf.nn.embedding_lookup(self.ent_embeddings, val_ents_array[:,0])
        # val_ents2 = tf.nn.embedding_lookup(self.ent_embeddings, val_ents_array[:,1])
        val_ents2 = tf.nn.embedding_lookup(self.ent_embeddings, val_ents2_index)

        print("val embed: {}/{}".format(val_ents1.shape, val_ents2.shape))

        val_embed1, val_embed2 = self.sess.run([val_ents1, val_ents2])

        del val_ents2_index
        gc.collect()
        return cal_threshold_via_embed(val_embed1, val_embed2)

        
        # val_index = np.array(range(1000))

        # hit_ratio_1 = 0
        # test_threshold = 0
        # flag = 0
        # for threshold in threshold_list:
        #     hit = 0
        #     total_num = 0
        #     cut_index = -1
        #     test_threshold = threshold

        #     for index in range(len(top_ents_list)):
        #         if ents_info[2] < threshold:
        #             cut_index = 
        #             if ents_info[0] == ents_info[1]:
        #                 hit += 1
        #     for i in range(len(val_index)):
        #         val_tmp = val_index[i]

        #         rank = np.argsort(-sim_matrix[i,:])

        #         if(sim_matrix[i][rank[0]] >= threshold):
        #             total_num +=1

        #             if rank[0] == val_tmp:
        #                 hit += 1
            
        #     if total_num > 0:
        #         hit_ratio_1 = round(hit / total_num,5)

        #         # print("hit@1: {} threshold: {}".format(round(hit / total_num, 3), round(threshold,3)))

        #         if hit_ratio_1 > 0.9:
        #             #print("hit@1: {} threshold: {}".format(round(hit / total_num, 3), round(threshold,3)))
        #             flag = 1
        #             return flag, hit_ratio_1, test_threshold
        #     else:
        #         return flag, hit_ratio_1, test_threshold

        # return flag, hit_ratio_1, test_threshold
    

            # print("total_num: ",total_num)

        # print("time: ",round(time.time() - s_t, 3))






    def test(self, aligned_test_pairs, test_index1, test_index2):
        s_t = time.time()
        test_ents_list = list(self.test_ents)
        #test_ents_list = list(self.ori_test_ents)
        print("Begin test: ",len(test_ents_list))
        test_ents_array = np.array(test_ents_list)
        test_ents1 = tf.nn.embedding_lookup(self.ent_embeddings, test_ents_array[:,0])
        test_ents2 = tf.nn.embedding_lookup(self.ent_embeddings, test_ents_array[:,1])

        test_embed1, test_embed2 = self.sess.run([test_ents1, test_ents2])
        
        print("ref1: ", np.array(test_embed1).shape)
        print("ref2: ", np.array(test_embed2).shape)

        evaluation(test_embed1, test_embed2, test_index1, test_index2, "Test", aligned_test_pairs)

        # self.aligned_ref_pairs = true_set

        # print("Now aligned ref pairs: {}".format(len(self.aligned_ref_pairs)))

        del test_embed1, test_embed2
        gc.collect()




    def get_kb1_embed(self):
        kb1_embed = tf.nn.embedding_lookup(self.ent_embeddings, self.kb_ents1)
        return self.sess.run(kb1_embed)

    def get_kb2_embed(self):
        kb2_embed = tf.nn.embedding_lookup(self.ent_embeddings, self.kb_ents2)
        return self.sess.run(kb2_embed)


