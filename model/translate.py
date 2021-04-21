from __future__ import division
import tensorflow as tf
import numpy as np
import os
from nltk.translate.bleu_score import sentence_bleu
from nmt_process import data_process
from new_attr_model import Seq2SeqModel
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# test_data_path = '../new_seq_test'
# train_data_path = '../new_seq_train'
# dict_path = '../des_val_vocab'
train_data_path = './data/zh_en_des'
dict_path = './data/whole_vocab_split'
# train_data_path = '../corpus.txt'
# dict_path = '../word_dict.tsv'
load_path = './pre_train/'
save_path = ''


num_epoches = 50
rnn_size = 128
num_layers = 2
embedding_size = 100
learning_rate = 0.001
use_attention = True
use_beam_search = True
beam_size = 5
max_gradient_norm = 5.0
cur_step = 0
display_step = 100
test_step = 300
use_checkpoint = False
top_k = [1,10,30,50]

def display_fuc(source, target, predict, id2word, go_int = 3, pad_int = 0, eos_int = 2, unk_int = 1):

	text_s = [id2word[idx] for idx in source if idx > 3] 
	text_t = [id2word[idx] for idx in target if idx > 3] 
	text_p = [id2word[idx] for idx in predict if idx > 3] 
	return text_s, text_t, text_p


train_data = data_process(train_data_path, dict_path, 200)
# test_data = data_process(test_data_path, dict_path, 200)

epoch_train_data = train_data.createbatchs()
# epoch_test_data = test_data.createbatchs()

print("Train data has %d batches." % (len(epoch_train_data)))

# print("Test data has %d batches." % (len(epoch_test_data)))

seq2seq_graph = tf.Graph()

tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

tf_config.gpu_options.allow_growth = True

with tf.Session(graph=seq2seq_graph,config=tf_config) as sess:
	model = Seq2SeqModel(sess = sess,
						rnn_size = rnn_size, 
						num_layers = num_layers,
						embedding_size = embedding_size, 
						learning_rate = learning_rate,
						word_to_idx = train_data.word2id, 
						use_attention = use_attention,
						beam_search = use_beam_search,
						beam_size = beam_size,
						max_gradient_norm = max_gradient_norm)

	if use_checkpoint:
		ckpt = tf.train.get_checkpoint_state(save_path)
		if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
			print('Reloading model parameters...')
			print(ckpt.model_checkpoint_path)
			model.saver.restore(sess, ckpt.model_checkpoint_path)
	else:
		print("Create new parameters...")
		sess.run(tf.global_variables_initializer())
	
	total_max_avg_bleu = 0
	min_loss = 100
	for epoch_num in range(num_epoches):
		print("-----------Epoch:{} / {}-----------".format(epoch_num+1, num_epoches))
		epoch_sum_loss = 0
		epoch_train_data = train_data.createbatchs()
		top_1 = 0
		top_10 = 0
		top_30 = 0
		for batch in epoch_train_data:
			cur_step = cur_step + 1
			#print("decode_len: ",np.max(batch.decode_len))
			# exit(0)
			s_t = time.time()
			loss, model_predict = model.train(batch)
			#loss = model.train(sess, batch)
			# top_1 += top_k_metric[0]
			# top_10 += top_k_metric[1]
			# top_30 += top_k_metric[2]

			epoch_sum_loss = epoch_sum_loss + round(loss,3)
			#print("rank: ", hit_1)
			# print("Cur_step: {} Loss: {} hits@: {} = {} time_cost: {}".format(cur_step, loss, top_k, top_k_metric,round(time.time() - s_t, 3)))
			# exit(0)
			if(cur_step > 0) & (cur_step % display_step == 0):
				print("Cur_step: {} Loss: {}".format(cur_step, loss))
				#print("Cur_step: {} Loss: {} hits@: {} = {} time_cost: {}".format(cur_step, loss, top_k, top_k_metric,round(time.time() - s_t, 3)))

		# top_1_ratio = round(top_1/61151, 6)
		# top_10_ratio = round(top_10/61151, 6)
		# top_30_ratio = round(top_30/61151, 6)
		# print("hit@1 = {} @10 = {} @30 = {}".format(top_1_ratio, top_10_ratio, top_30_ratio))
				#exit(0)
				# print("encode: {} decode: {} predict: "%(np.array(batch.encode_inputs).shape, np.array(batch.decode_inputs).shape, np.array(predict).shape))
				# print(np.array(batch.encode_inputs).shape)
				# print(np.array(batch.decode_inputs).shape)
				# print(np.array(predict).shape)

			# 	new_predict = [[] for _ in range(np.shape(model_predict)[0])]
			# 	for a in range(np.shape(model_predict)[0]):
			# 		for b in range(np.shape(model_predict)[1]):
			# 			new_predict[a].append(model_predict[a][b][0])

			# 	for source, target, predict in zip(batch.encode_inputs[:3], batch.decode_inputs_in[:3], new_predict[:3]):
			# 		text_source, text_target, text_predict = display_fuc(source, target, predict, train_data.id2word)
			# 		print('-------------------')
			# 		print("Source: ", ' '.join(text_source))
			# 		print("Target: ",' '.join(text_target))
			# 		print("Beam-5 Predict: ",' '.join(text_predict)) 
			# 		print('-------------------')


			# if(cur_step > 0) & (cur_step % test_step ==0):
			# 	print('------------Test: cur_step:{}-----------'.format(cur_step))
			# 	epoch_test_data = test_data.createbatchs()
			# 	test_num = 0
			# 	max_bleu = 0
			# 	sum_bleu = 0
			# 	for test_batch in epoch_test_data:
			# 		test_num += 1

			# 		model_predict = model.inference(test_batch)

			# 		new_predict = [[] for _ in range(np.shape(model_predict)[0])]
			# 		for a in range(np.shape(model_predict)[0]):
			# 			for b in range(np.shape(model_predict)[1]):
			# 				new_predict[a].append(model_predict[a][b][0])
			# 		#for source, target, predict in zip(batch.encode_inputs[:5], batch.decode_inputs_in[:5], new_predict[:5]):
			# 		for index in range(len(new_predict)):
			# 			source = test_batch.encode_inputs[index]
			# 			target = test_batch.decode_inputs_in[index]
			# 			predict = new_predict[index]
			# 			text_source, text_target, text_predict = display_fuc(source, target, predict, train_data.id2word)

			# 			bleu_score = round(sentence_bleu([text_target], text_predict, weights=(1,0)),3)

			# 			sum_bleu = sum_bleu + bleu_score

			# 			if (test_num % 9 == 0) and (index < 5):
			# 				print('-------------------')
			# 				print("Source: ", ' '.join(text_source))
			# 				print("Target: ",' '.join(text_target))
			# 				print("Beam-5 Predict: ",' '.join(text_predict)) 
			# 				print('-------------------')
			# 	avg_bleu_score = round(sum_bleu/5000,3) *100
			# 	if avg_bleu_score > total_max_avg_bleu:
			# 		total_max_avg_bleu = avg_bleu_score
			# 		model.saver.save(sess, save_path + 'new_des_pre.ckpt',global_step = epoch_num)
			# 	print("Avg_bleu_score: {} Total_max_avg_bleu: {}".format(avg_bleu_score, total_max_avg_bleu))

		#epoch_avg_loss = round(epoch_sum_loss / 306,6)
		epoch_avg_loss = round(epoch_sum_loss / len(epoch_train_data),6)
		if(epoch_avg_loss < min_loss):
			min_loss = epoch_avg_loss
			model.saver.save(sess, load_path + 'des_pre.ckpt',global_step = epoch_num)
			#self.saver.save(session, self.save_path + 'rl_wc.ckpt', global_step=current_step)

	print("Total avg min loss: {}".format(min_loss))
	#print("Total avg min loss: {}   Max_bleu_score: {}".format(min_loss, total_max_avg_bleu))


			# 		ans_prediction = sess.run([inference_predict], feed_dict=feed_dict)[0]

   #  new_ans_prediction = [[] for _ in range(np.shape(ans_prediction)[0])]
   #  for a in range(np.shape(ans_prediction)[0]):
   #      for b in range(np.shape(ans_prediction)[1]):
   #          new_ans_prediction[a].append(ans_prediction[a][b][0])

   #  ans_prediction = new_ans_prediction
   #  print(new_ans_prediction[0])
   #  print(np.shape(new_ans_prediction))

   #  pad_int = word2idx['<PAD>']
   #  eos_int = word2idx['<EOS>']

   #  for rs_ask, rs_ans in zip(ask_data, ans_prediction):
   #      ask_seq = " ".join(utils.seq2text(
   #          rs_ask, idx2word, pad_int, eos_int, True))
   #      ans_seq = " ".join(utils.seq2text(
   #          rs_ans, idx2word, pad_int, eos_int, True))
   #      print("ask: %s\nans: %s\n" % (ask_seq, ans_seq))




