import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import embed_sequence
from tensorflow.python.util import nest


import tensorflow as tf
from tensorflow.python.util import nest
import gc

class Seq2SeqModel():
	def __init__(self, sess, rnn_size, num_layers, embedding_size, learning_rate, word_to_idx, use_attention, beam_search, beam_size, max_gradient_norm=5.0, encode_bi = True):

		self.sess = sess
		self.learing_rate = learning_rate
		self.embedding_size = embedding_size
		self.rnn_size = rnn_size
		self.num_layers = num_layers
		self.word_to_idx = word_to_idx
		self.vocab_size = len(self.word_to_idx)
		self.use_attention = use_attention
		self.beam_search = beam_search
		self.beam_size = beam_size
		self.max_gradient_norm = max_gradient_norm
		self.encode_bi = encode_bi
		self.build_model()

	def _create_rnn_cell(self,rnn_size):
		def single_rnn_cell():
			single_cell = tf.contrib.rnn.LSTMCell(rnn_size)
			cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=self.keep_prob_placeholder)
			return cell

		cell = tf.contrib.rnn.MultiRNNCell([single_rnn_cell() for _ in range(self.num_layers)])	
		
		return cell

	def build_model(self):
		print('building model-------------------------')

		self.encoder_inputs = tf.placeholder(tf.int32, [None, None], name='encoder_inputs')
		self.encoder_inputs_length = tf.placeholder(tf.int32, [None], name='encoder_inputs_length')

		self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
		self.keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob_placeholder')

		self.decoder_targets = tf.placeholder(tf.int32, [None, None], name='decoder_targets')
		self.decoder_targets_count = tf.placeholder(tf.int32, [None, None], name='decoder_targets_count')
		self.decoder_targets_length = tf.placeholder(tf.int32, [None], name='decoder_targets_length')
		self.max_target_sequence_length = tf.reduce_max(self.decoder_targets_length, name='max_target_len')
		self.mask = tf.sequence_mask(self.decoder_targets_length, self.max_target_sequence_length, dtype=tf.float32, name='masks')
	




		print("building encoder model---------------")		
		with tf.variable_scope('encode'):
			
			self.embedding = tf.Variable(tf.random_normal([self.vocab_size, self.embedding_size]), name="embedding", dtype=tf.float32)
			# self.target_seq_embed = tf.nn.embedding_lookup(self.embedding, self.target_seq)
			# self.predict_seq_embed = tf.nn.embedding_lookup(self.embedding, self.predict_seq)
			encoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding, self.encoder_inputs)
			if self.encode_bi:
				print("use bi-listm")
				enc_cell_fw = self._create_rnn_cell(self.rnn_size)
				enc_cell_bw = self._create_rnn_cell(self.rnn_size)
				encoder_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(enc_cell_fw, enc_cell_bw, encoder_inputs_embedded, sequence_length=self.encoder_inputs_length, dtype=tf.float32)

				encoder_outputs = tf.concat([encoder_outputs[0], encoder_outputs[1]], -1)

				new_enc_state = []
				for i in range(len(encoder_state[0])):
					new_enc_state_c = tf.concat([encoder_state[0][i].c, encoder_state[1][i].c], -1)
					new_enc_state_h = tf.concat([encoder_state[0][i].h, encoder_state[1][i].h], -1)
					new_enc_state.append(tf.contrib.rnn.LSTMStateTuple(c=new_enc_state_c, h=new_enc_state_h))

				encoder_state = tuple(new_enc_state)

			else:
				encoder_cell = self._create_rnn_cell(self.rnn_size)
				encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_inputs_embedded,sequence_length=self.encoder_inputs_length,dtype=tf.float32)
			# trainable_params = tf.trainable_variables()
			# print(len(trainable_params))
			# for v in trainable_params:
			# 	print(v)


		# build evaluate model:
		self.target_seq = tf.placeholder(tf.int32, [None, None], name='target_seq')
		self.predict_seq = tf.placeholder(tf.int32, [None, None], name='predict_seq')
		self.target_seq_length = tf.placeholder(tf.int32, [None], name='target_length')
		self.predict_seq_length = tf.placeholder(tf.int32, [None], name='predict_length')
		
		self.max_target_seq_length = tf.shape(self.target_seq)[1]
		self.max_predict_seq_length = tf.shape(self.predict_seq)[1]

		self.target_seq_embed = tf.nn.embedding_lookup(self.embedding, self.target_seq)
		self.predict_seq_embed = tf.nn.embedding_lookup(self.embedding, self.predict_seq)

		self.target_seq_mask = tf.expand_dims(tf.sequence_mask(self.target_seq_length, self.max_target_seq_length, dtype=tf.float32), -1)
		self.predict_seq_mask = tf.expand_dims(tf.sequence_mask(self.predict_seq_length, self.max_predict_seq_length, dtype = tf.float32), -1)

		self.re_target_embedding = self.target_seq_embed * self.target_seq_mask
		self.re_predict_embedding = self.predict_seq_embed * self.predict_seq_mask

		self.target_sum_embed =  tf.nn.l2_normalize(tf.reduce_sum(self.re_target_embedding, 1) ,1)
		self.predict_sum_embed =  tf.nn.l2_normalize(tf.reduce_sum(self.re_predict_embedding, 1),1)

		#
		print("building decoder model---------------")
		with tf.variable_scope('decode'):
			encoder_inputs_length = self.encoder_inputs_length
			
			with tf.variable_scope('shared_attention_mechanism'):
				#attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=self.rnn_size, memory=encoder_outputs, memory_sequence_length=encoder_inputs_length)
				attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.rnn_size*2, memory=encoder_outputs, memory_sequence_length=encoder_inputs_length)


			decoder_cell = self._create_rnn_cell(self.rnn_size*2)
			decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell, attention_mechanism=attention_mechanism,attention_layer_size=self.rnn_size*2, name='Attention_Wrapper')
			


			batch_size = self.batch_size
			decoder_initial_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32).clone(cell_state=encoder_state)

			
			output_layer = tf.layers.Dense(self.vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
			# Train
			# ending = tf.strided_slice(self.decoder_targets, [0, 0], [self.batch_size, -1], [1, 1])
			# decoder_input = tf.concat([tf.fill([self.batch_size, 1], self.word_to_idx['<GO>']), ending], 1)

			decoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding, self.decoder_targets)
			training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_inputs_embedded,
																sequence_length=self.decoder_targets_length,
																time_major=False, 
																name='training_helper')

			training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, 
																helper=training_helper,
																initial_state=decoder_initial_state,
																output_layer=output_layer)
			# training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, 
			# 													helper=training_helper,
			# 													initial_state=decoder_initial_state)

			with tf.variable_scope('decode_with_shared_attention'):
				decoder_outputs, f_s, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder,
																		impute_finished=False,
																		maximum_iterations=self.max_target_sequence_length)
			# self.decode_out = f_s
			# self.decoder_logits = tf.layers.Dense(decoder_outputs.rnn_output, self.vocab_size)
   #    		self.decoder_predict = tf.argmax(self.decoder_logits, 2)
			self.decoder_logits_train = tf.identity(decoder_outputs.rnn_output)
			#self.decoder_predict_train = tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_pred_train')
			self.decoder_predict_train = tf.identity(decoder_outputs.sample_id)



			self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.decoder_logits_train,
														targets=self.decoder_targets_count,
														weights=self.mask)

			optimizer = tf.train.AdamOptimizer()
			trainable_params = tf.trainable_variables()
			# print(len(trainable_params))
			# for v in trainable_params:
			# 	print(v)
			gradients = tf.gradients(self.loss, trainable_params)
			clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
			self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))
			# gradients = optimizer.compute_gradients(self.loss)
			# copped_gradients = [(tf.clip_by_value(grad, -5.0, 5.0), var)
			# 					for grad, var in gradients if grad is not None]
			# self.train_op = optimizer.apply_gradients(copped_gradients)


			# inference:
			start_tokens = tf.ones([self.batch_size, ], tf.int32) * self.word_to_idx['<GO>']
			end_token = self.word_to_idx['<EOS>']


			if self.beam_search:
				#if self.beam_search:
				print("use beamsearch decoding..")
				encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=self.beam_size)
				encoder_state = nest.map_structure(lambda s: tf.contrib.seq2seq.tile_batch(s, self.beam_size), encoder_state)
				#encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier = self.beam_size)
				encoder_inputs_length = tf.contrib.seq2seq.tile_batch(self.encoder_inputs_length, multiplier=self.beam_size)
				

				with tf.variable_scope('shared_attention_mechanism', reuse = True):
					#attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=self.rnn_size, memory=encoder_outputs, memory_sequence_length=encoder_inputs_length)
					attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.rnn_size*2, memory=encoder_outputs, memory_sequence_length=encoder_inputs_length)

				decoder_cell = self._create_rnn_cell(self.rnn_size*2)
				decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell, attention_mechanism=attention_mechanism,attention_layer_size=self.rnn_size*2, name='Attention_Wrapper')
				decoder_initial_state = decoder_cell.zero_state(self.batch_size*self.beam_size, tf.float32).clone(cell_state=encoder_state)

				inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=decoder_cell,
																		embedding= self.embedding,
																		start_tokens=start_tokens, end_token=end_token,
																		initial_state=decoder_initial_state,
																		beam_width=self.beam_size,
																		output_layer=output_layer)
			else:
				print("Use Greedy Searching")
				decoder_initial_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32).clone(cell_state=encoder_state)
				decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding = self.embedding,
																			start_tokens=start_tokens,
																			end_token=end_token)
				inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=decoding_helper,
																	initial_state=decoder_initial_state,
																	output_layer=output_layer)

			with tf.variable_scope('decode_with_shared_attention', reuse = True):
				decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder,maximum_iterations = self.max_target_sequence_length)
			
				#self.beam_logits = decoder_outputs.beam_search_decoder_output
			if self.beam_search:
				self.decoder_predict_decode = decoder_outputs.predicted_ids

			else:
				self.decoder_predict_decode = tf.expand_dims(decoder_outputs.sample_id, -1)

			self.saver = tf.train.Saver(max_to_keep=3)
			# rainable_params = tf.global_variables()
			# print(len(rainable_params))
			# for v in rainable_params:
			# 	print(v)
	
	# def get_target_embedding(self, batch, model_predict):
	# 	decoder_embedded_op = tf.nn.embedding_lookup(self.embedding, self.decoder_targets)
	# 	self.sess.run(decoder_embedded_op)
	# 	return decoder_embedded


	def eval_score(self, batch, model_predict):
		new_predict = [[] for _ in range(np.shape(model_predict)[0])]
		for a in range(np.shape(model_predict)[0]):
			for b in range(np.shape(model_predict)[1]):
				new_predict[a].append(model_predict[a][b][0])

		decode_length = batch.decode_len
		decode_input = batch.decode_inputs_out

		input_target_len = np.array(decode_length) - 1

		input_predict_len = []
		for i in range(len(new_predict)):
			seq_list = [idx for idx in new_predict[i] if (idx > 3) or (idx == 0)]
			input_predict_len.append(len(seq_list))
		# print(decode_input)
		# print(input_target_len)
		# print(new_predict)
		# print(input_predict_len)
		# print('---')
		feed_dict = {
					self.target_seq: decode_input, 
					self.predict_seq: new_predict,
					self.target_seq_length: input_target_len,
					self.predict_seq_length: input_predict_len
		}
		target_embed, predict_embed = self.sess.run([self.target_sum_embed, self.predict_sum_embed], feed_dict=feed_dict)

		score_matrix = np.matmul(target_embed, predict_embed.T)


		top_k = [1,10,30,50]
		top_k_matric = np.array([0 for k in top_k])
		for i in range(len(score_matrix)):
			rank = np.argsort(-score_matrix[i,:])

			true_index = np.where(rank == i)[0][0]

			for k in range(len(top_k)):
				if true_index < top_k[k]:
					top_k_matric[k] += 1	
		del score_matrix
		del target_embed
		del predict_embed
		del new_predict
		gc.collect()

		return top_k_matric
		# print('hits@{} = {}'.format(top_k, top_k_matric))


	def pro_embed(self):
		# with self.sess.as_default():
		self.sess.run(self.embedding)

			#self.sess.run(tf.global_variables_initializer())


	def train(self, batch):
		feed_dict = {self.encoder_inputs: batch.encode_inputs,
					self.encoder_inputs_length: batch.encode_len,
					self.decoder_targets: batch.decode_inputs_in,
					self.decoder_targets_count: batch.decode_inputs_out,
					self.decoder_targets_length: batch.decode_len,
					self.keep_prob_placeholder: 0.5,
					self.batch_size: len(batch.encode_inputs)}
		_, loss ,pre = self.sess.run([self.train_op, self.loss, self.decoder_predict_decode], feed_dict=feed_dict)
		# e_s= self.sess.run(self.e_s, feed_dict=feed_dict)
		# logits,truth, mask, loss, beam_logits = sess.run([self.decoder_logits_train, self.decoder_targets_count, self.mask, self.loss,self.beam_logits], feed_dict = feed_dict)

		# top_k_matric = self.eval_score(batch, pre)
		# print("logits: ",logits)
		# print("truth: ",truth)
		# print("mask: ",mask)
		# print("loss: ",loss)
		# print("beam logits: ",beam_logits)
		# exit(0)
		return loss, pre
		# logits , label_in ,label_out,ml= sess.run([self.decoder_logits_train, self.decoder_targets,self.decoder_targets_count,self.max_target_sequence_length], feed_dict=feed_dict)
		# print("1: ",len(logits[0]))
		# print("in: ",len(label_in[0]))
		# print("out: ",len(label_out[0]))
		# print(ml)
		# print(np.max(batch.decode_len))
		#return loss
		# print(loss)
		# pre = sess.run(self.decoder_predict_decode,feed_dict = feed_dict)
		# print(np.array(pre).shape)


	def inference(self, batch):

		feed_dict = {self.encoder_inputs: batch.encode_inputs,
					self.encoder_inputs_length: batch.encode_len,
					self.decoder_targets: batch.decode_inputs_in,
					self.decoder_targets_count: batch.decode_inputs_out,
					self.decoder_targets_length: batch.decode_len,
					self.keep_prob_placeholder: 0.5,
					self.batch_size: len(batch.encode_inputs)}

		pre = self.sess.run(self.decoder_predict_decode, feed_dict = feed_dict)

		return pre









