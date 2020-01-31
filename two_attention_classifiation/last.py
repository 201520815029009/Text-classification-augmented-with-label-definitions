import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer, xavier_initializer
from attention_gru_cell import AttentionGRUCell


class QaCNN(object):
    def __init__(self, trainable_embeddings, question_length, answer_length, embed_size,
                 vocab_size, learning_rate, decay_steps, decay_rate):
        self.accusation_num_classes = 202
        self.trainable_embeddings = trainable_embeddings
        self.question_length = question_length
        self.answer_length = answer_length
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * 0.6)
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.decay_steps, self.decay_rate = decay_steps, decay_rate
        self.num_hops = 3

        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))

        self.num_filters = embed_size
        self.window_size = 5  #5

        self.W_conv2 = weight_variable('W_conv2', [self.window_size, self.embed_size, 1, self.num_filters])
        self.b_conv2 = bias_variable('b_conv2', [self.num_filters])

        self.W1 = tf.Variable(tf.random_uniform([self.embed_size, self.embed_size], 0.0, 1.0), name="attention_W1")
        self.W2 = tf.Variable(tf.random_uniform([1, self.embed_size], 0.0, 1.0), name="attention_W2")

        self.add_placeholder()

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.embeddings_weight = tf.get_variable("embeddings", [self.vocab_size, self.embed_size], dtype=tf.float32,
                                                     initializer=tf.random_normal_initializer(stddev=0.1),
                                                     trainable=self.trainable_embeddings)
            self.embeddings_question = tf.nn.embedding_lookup(self.embeddings_weight, self.input_question)
            self.embeddings_answer = tf.nn.embedding_lookup(self.embeddings_weight, self.input_answer)

        raw_representation_question, q_word = self.get_gru()
        raw_representation_answer, a_word = self.get_input_representation()

        alphalist = []
        with tf.variable_scope("memory", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> build episodic memory')
            # generate n_hops episodes
            prev_memory = raw_representation_question  # probability
            for i in range(self.num_hops):
                # get a new episode
                print('==> generating episode', i)
                episode, alpha = self.generate_episode(prev_memory, raw_representation_question, raw_representation_answer, i)
                alphalist.append(alpha[0])
                # untied weights for memory update
                with tf.variable_scope("hop_%d" % i):
                    prev_memory = tf.layers.dense(tf.concat([prev_memory, episode, raw_representation_question], 1),
                                                  self.embed_size,
                                                  activation=tf.nn.relu)
        self.alpha = alphalist 
        #print(self.alpha) 
        articles = tf.unstack(a_word, axis=1)
        fact = []
        word_att = []
        gru_cell = tf.contrib.rnn.GRUCell(self.embed_size)
        for i, article in enumerate(articles):
            fact_new, att = self.compute(q_word, article)
            fact.append(fact_new)
            word_att.append(att[0])   #[202, b, s,s']
        self.word_att= word_att[196]
        # print(self.word_att)
        fact = tf.transpose(fact, [1,0,2,3])
        fact = tf.multiply(fact, tf.expand_dims(tf.expand_dims(alpha, -1), -1))
        fact_new = tf.reduce_max(fact, axis=1)
        with tf.variable_scope("word-gru"):
            _, q_vec = tf.nn.dynamic_rnn(gru_cell, fact_new, dtype=np.float32, sequence_length=self.input_question_len)

        output = tf.layers.dense(tf.concat([q_vec, raw_representation_question], 1),self.embed_size, activation=tf.nn.relu)

        # pass memory module output through linear answer module
        with tf.variable_scope("answer", initializer=tf.contrib.layers.xavier_initializer()):
            self.output = self.add_answer_module(output, raw_representation_question, prev_memory)

        self.pred = self.get_predictions(self.output)
        self.calculate_loss = self.add_loss_op(self.output)
        self.train_step = self.add_training_op(self.calculate_loss)

    def add_placeholder(self):
        self.input_question = tf.placeholder(tf.int32, [None, self.question_length])
        self.input_question_len = tf.placeholder(tf.int32, [None, ])
        self.input_answer = tf.placeholder(tf.int32, [None, 202, self.answer_length])
        self.input_y_accusation = tf.placeholder(tf.float32, [None, self.accusation_num_classes],
                                                 name="input_y_accusation")
        self.dropout_keep_prob = tf.placeholder(tf.float32)

    def get_gru(self):
        #input_q = self.cnn_representation_raw(self.embeddings_question, self.question_length, self.W_conv1, self.b_conv1)
        
        #gru_cell_fw = tf.contrib.rnn.GRUCell(self.embed_size)
        gru_cell_bw = tf.contrib.rnn.GRUCell(self.embed_size)
        output, q_vec = tf.nn.dynamic_rnn(gru_cell_bw, self.embeddings_question, dtype=np.float32, sequence_length=self.input_question_len)
        #output = out1+out2

        output_re = tf.reshape(tf.transpose(output, [2, 0, 1]), [self.embed_size, -1])
        alpha = tf.matmul(self.W2, tf.nn.tanh(tf.matmul(self.W1, output_re)))
        alpha = tf.transpose(alpha)
        alpha = tf.nn.softmax(tf.reshape(alpha, [-1, self.question_length]))
        output = tf.multiply(output, tf.expand_dims(alpha, -1))
        q_vec = tf.reduce_sum(output, axis=1)
        return q_vec, output


    def cnn_representation_raw(self, item, length, w, b):
        # we need to add another dimension, because cnn works on 3d data only
        cnn_input = tf.expand_dims(item, -1)
        convoluted = tf.nn.bias_add(
            tf.nn.conv2d(
                cnn_input,
                w,
                strides=[1, 1, self.embed_size, 1],
                padding='SAME'
            ),
            b
        )
        cnn_output = tf.reshape(convoluted, [-1, length, self.num_filters])
        return cnn_output


    def get_input_representation(self):
        #inputs = tf.reduce_sum(self.embeddings_answer, 2)

        article_w = []
        article_se = []

        for i, fv in enumerate(tf.unstack(self.embeddings_answer, axis=1)):
            word = self.cnn_representation_raw(fv, self.answer_length, self.W_conv2, self.b_conv2)
            semantic = tf.reduce_sum(word, axis=1, keep_dims=False)
            article_w.append(word)
            article_se.append(semantic)

        word = tf.transpose(article_w, [1, 0, 2, 3])
        semantic = tf.transpose(article_se, [1, 0, 2])

        # apply dropout
        fact_vecs = tf.nn.dropout(semantic, self.dropout_keep_prob)
        return fact_vecs, word

    def get_attention(self, q_vec, prev_memory, fact_vec, reuse):
        """Use question vector and previous memory to create scalar attention for current fact"""
        with tf.variable_scope("attention", reuse=reuse):
            features = [fact_vec * q_vec,
                        fact_vec * prev_memory,
                        tf.abs(fact_vec - q_vec),
                        tf.abs(fact_vec - prev_memory)]

            feature_vec = tf.concat(features, 1)

            attention = tf.contrib.layers.fully_connected(feature_vec, self.embed_size, activation_fn=tf.nn.tanh,
                                                          reuse=reuse, scope="fc1")

            attention = tf.contrib.layers.fully_connected(attention, 1, activation_fn=None, reuse=reuse, scope="fc2")

        return attention

    def generate_episode(self, memory, q_vec, fact_vecs, hop_index):
        """Generate episode by applying attention to current fact vectors through a modified GRU"""

        attentions = [tf.squeeze(
            self.get_attention(q_vec, memory, fv, bool(hop_index) or bool(i)), axis=1)
            for i, fv in enumerate(tf.unstack(fact_vecs, axis=1))]

        attentions = tf.transpose(attentions)
        attentions = tf.nn.softmax(attentions)
        attention = attentions / tf.expand_dims(tf.reduce_sum(attentions, axis=1), axis=-1)  # [b, 202]
        attentions = tf.expand_dims(attention, axis=-1)

        gru_inputs = tf.multiply(attentions, fact_vecs)
        episode = tf.reduce_max(gru_inputs, axis=1)

        return episode, attention

    def compute(self, fact, article):
        # fact = [B, s, d]   article = [B, s', d]
        shuffled = tf.transpose(article, perm=[0, 2, 1])  # B x D x Q
        inter = tf.matmul(fact, shuffled)
        alphas_r = tf.nn.softmax(inter)  # element-wise
        alphas_r = alphas_r / tf.expand_dims(tf.reduce_sum(alphas_r, axis=2), axis=-1)  # B x s x s'
        q_rep = tf.matmul(alphas_r, article)  # B x N x D   # matrix multiply
        return q_rep, alphas_r   # tf.multiply(fact, q_rep)

    def add_answer_module(self, rnn_output, q_vec, prev_memory):
        """Linear softmax answer module"""
        rnn_output = tf.nn.dropout(rnn_output, self.dropout_keep_prob)
        #prev_memory= tf.nn.dropout(prev_memory, self.dropout_keep_prob)
        #output = tf.layers.dense(tf.concat([rnn_output, q_vec, q_vec, prev_memory], 1), 202, activation=None)
        output = tf.layers.dense(tf.concat([rnn_output, q_vec, prev_memory], 1), 202, activation=None)
        #output = tf.layers.dense(rnn_output, 202, activation=None)
        return output

    def get_predictions(self, output):
        preds = tf.nn.softmax(output)
        pred = tf.argmax(preds, 1)
        return pred

    def add_loss_op(self, output):
        """Calculate loss"""
        loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_accusation, logits=output))
        # add l2 regularization for all variables except biases
        for v in tf.trainable_variables():
            if not 'bias' in v.name.lower():
                loss += 0.001 * tf.nn.l2_loss(v)
        return loss

    def add_training_op(self, loss):
        """Calculate and apply gradients"""
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        gvs = opt.compute_gradients(loss)
        train_op = opt.apply_gradients(gvs)
        return train_op

def weight_variable(name, shape, regularization=None):
    regularizer = None
    if regularization is not None:
        regularizer = l2_regularizer(1e-5)
    return tf.get_variable(name, shape=shape, initializer=xavier_initializer(), regularizer=regularizer)


def bias_variable(name, shape, value=0.1):
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(value))