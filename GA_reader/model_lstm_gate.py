#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorflow.contrib.layers import l2_regularizer, xavier_initializer
from utils.model_helper import *
from tensorflow.contrib import rnn

class GAReader:
    def __init__(self, vocab_size, gru_size, batch_size, embed_dim, train_emb, gating_fn, init_learning_rate,
                 max_seq_len, max_article_len):
        self.embed_dim = embed_dim
        self.lstm_cell_size = embed_dim
        self.lw_cell_size = embed_dim
        self.batch_size = batch_size
        self.train_emb = train_emb
        self.gating_fn = gating_fn
        self.n_vocab = vocab_size
        self.lr = tf.Variable(init_learning_rate, trainable=False, name="learning_rate")
        self.learning_rate_decay_half_op = tf.assign(self.lr, self.lr * 0.5)

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))

        self.lstm_cell_size = 64
        self.lw_cell_size = 64
        self.num_filters = 128
        self.window_size = 3
        self.W_conv1 = weight_variable('W_conv', [self.window_size, self.embed_dim, 1, self.num_filters])
        self.b_conv1 = bias_variable('b_conv', [self.num_filters])
        self.__lstm_history = set()
        self.initialize_weights()

        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.doc = tf.placeholder(tf.int32, [None, max_seq_len], name="doc")
        self.pos = tf.placeholder(tf.int32, [None, max_article_len], name="pos_charge")
        self.neg = tf.placeholder(tf.int32, [None, max_article_len], name="neg_charge")

        # word embedding
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.word_embedding = tf.get_variable("word_embedding", [self.n_vocab, self.embed_dim], dtype=tf.float32,
                                                initializer=tf.random_normal_initializer(stddev=0.1), trainable=self.train_emb)
            self.embeddings_question = tf.nn.embedding_lookup(self.word_embedding, self.doc)
            self.embeddings_answer_good = tf.nn.embedding_lookup(self.word_embedding, self.pos)
            self.embeddings_answer_bad = tf.nn.embedding_lookup(self.word_embedding, self.neg)

        raw_representation_question = self.bilstm_representation_raw(self.embeddings_question, self.doc, re_use_lstm=False)
        raw_representation_answer_good = self.bilstm_representation_raw(self.embeddings_answer_good, self.pos, re_use_lstm=True)
        raw_representation_answer_bad = self.bilstm_representation_raw(self.embeddings_answer_bad, self.neg, re_use_lstm=True)


        self.question_good_pooling_weight, self.answer_good_pooling_weight = \
            self._compute(raw_representation_question, raw_representation_answer_good, tf.cast(self.doc, tf.bool),
                          tf.cast(self.pos, tf.bool), self.pos, self.doc, None)

        self.question_bad_pooling_weight, self.answer_bad_pooling_weight = \
            self._compute(raw_representation_question, raw_representation_answer_bad, tf.cast(self.doc, tf.bool),
                          tf.cast(self.neg, tf.bool), self.neg, self.doc, True)

        self.create_outputs(
            self.question_good_pooling_weight, self.answer_good_pooling_weight,
            self.question_bad_pooling_weight, self.answer_bad_pooling_weight
        )


    def bilstm_representation_raw(self, item, indices, re_use_lstm, name='lstm'):
        tensor_non_zero_token = tf.ceil(tf.to_float(indices) / tf.reduce_max(tf.to_float(indices), [1], keep_dims=True))
        sequence_length = tf.to_int64(tf.reduce_sum(tensor_non_zero_token, 1))
        with tf.variable_scope(name, reuse=re_use_lstm):
            (fw_doc_states, bk_doc_states), _ = tf.nn.bidirectional_dynamic_rnn(
                self.lstm_cell_forward,
                self.lstm_cell_backward,
                item,
                dtype=tf.float32,
                sequence_length=sequence_length
            )
            return tf.concat([fw_doc_states, bk_doc_states], axis=2)

    def _compute(self, doc_bi_embed, qry_bi_embed, mask_doc, mask_art, qry, doc, reuse):
        # soft attention
        with tf.variable_scope("pool-filter3", reuse=reuse):
            # attention_qry_bi_embed = attentive_pooling_weights(self.U_AP, self.b, qry_bi_embed, qry)
            # print('attention_article')  # B x S x 128
            # print(attention_qry_bi_embed)

            inter = pairwise_interaction(doc_bi_embed, qry_bi_embed)
            doc_embed = gated_attention(doc_bi_embed, qry_bi_embed, inter, mask_art, gating_fn=self.gating_fn)
            print('doc and article interactive')
            print(doc_embed)
            # self attention
            # inter = pairwise_interaction(doc_embed, doc_embed)
            # match = gated_attention(doc_embed, doc_embed, inter, mask_doc, gating_fn=self.gating_fn)

            attention_doc_embed = self.positional_weighting(
                doc_embed,
                doc,
                'answer'
            )
            print('doc self attention')
            print(attention_doc_embed)

            doc = weighted_pooling(doc_embed, attention_doc_embed, doc)
            print(doc)
            article = tf.reduce_max(qry_bi_embed, [1], keep_dims=False)
            print(article)

            # doc = tf.reduce_max(attention_doc_embed, [1], keep_dims=False)
            # article = tf.reduce_max(attention_qry_bi_embed, [1], keep_dims=False)

            # Maxpooling over the outputs
            # doc_match = tf.expand_dims(doc_embed, -1)
            # article = tf.expand_dims(qry_bi_embed, -1)
            # doc = tf.reduce_max(doc_match, [1], keep_dims=False)
            # article = tf.reduce_max(article, [1], keep_dims=False)
            # doc = self.maxpool_tanh(doc_embed, doc)
            # article = self.maxpool_tanh(qry_bi_embed, qry)
        return doc, article

    def initialize_weights(self):
        with tf.variable_scope('lstm_cell_fw'):
            self.lstm_cell_forward = rnn.BasicLSTMCell(self.lstm_cell_size, state_is_tuple=True)
        with tf.variable_scope('lstm_cell_bw'):
            self.lstm_cell_backward = rnn.BasicLSTMCell(self.lstm_cell_size, state_is_tuple=True)

        self.U_AP = weight_variable('U_AP', [self.lstm_cell_size * 2, self.lstm_cell_size * 2])
        self.b = weight_variable('b', [self.lstm_cell_size * 2])

        cell_size = self.lw_cell_size
        self.dense_weighting_Q = weight_variable('dense_weighting_Q', [cell_size + cell_size, 1])
        self.dense_weighting_A = weight_variable('dense_weighting_A', [cell_size + cell_size, 1])

        with tf.variable_scope('lstm_cell_weighting_Q_fw'):
            self.lstm_cell_weighting_Q_fw = rnn.BasicLSTMCell(cell_size, state_is_tuple=True)

        with tf.variable_scope('lstm_cell_weighting_Q_bw'):
            self.lstm_cell_weighting_Q_bw = rnn.BasicLSTMCell(cell_size, state_is_tuple=True)

        with tf.variable_scope('lstm_cell_weighting_A_fw'):
            self.lstm_cell_weighting_A_fw = rnn.BasicLSTMCell(cell_size, state_is_tuple=True)

        with tf.variable_scope('lstm_cell_weighting_A_bw'):
            self.lstm_cell_weighting_A_bw = rnn.BasicLSTMCell(cell_size, state_is_tuple=True)

    def positional_weighting(self, raw_representation, indices, item_type):
        re_use = item_type in self.__lstm_history
        self.__lstm_history.add(item_type)
        if item_type == 'question':
            lstm_cell_fw = self.lstm_cell_weighting_Q_fw
            lstm_cell_bw = self.lstm_cell_weighting_Q_bw
            dense_weight = self.dense_weighting_Q
        else:
            lstm_cell_fw = self.lstm_cell_weighting_A_fw
            lstm_cell_bw = self.lstm_cell_weighting_A_bw
            dense_weight = self.dense_weighting_A

        tensor_non_zero_token = non_zero_tokens(tf.to_float(indices))
        sequence_length = tf.to_int64(tf.reduce_sum(tensor_non_zero_token, 1))

        with tf.variable_scope('lstm_{}'.format(item_type), reuse=re_use):
            (fw_doc_states, bk_doc_states), _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell_fw,
                lstm_cell_bw,
                raw_representation,
                dtype=tf.float32,
                sequence_length=sequence_length
            )
            lstm_output = tf.concat([fw_doc_states, bk_doc_states], axis=2)

        # apply dense over each individual lstm output
        flat_lstm_output = tf.reshape(lstm_output, [-1, self.lw_cell_size + self.lw_cell_size])
        dense_mul_flat = tf.matmul(flat_lstm_output, dense_weight)
        h1_layer = tf.reshape(dense_mul_flat, [-1, tf.shape(raw_representation)[1]])

        return attention_softmax(h1_layer, tensor_non_zero_token)

    def create_outputs(self, question_good, answer_good, question_bad, answer_bad):
        similarity_type = 'cosine'
        if similarity_type == 'gesd':
            similarity = gesd_similarity
        else:
            similarity = cosine_similarity

        # We apply dropout before similarity. This only works when we dropout the same indices in question and answer.
        # Otherwise, the similarity would be heavily biased (in case of angular/cosine distance).
        dropout_multiplicators = tf.nn.dropout(question_good * 0.0 + 1.0, self.keep_prob)

        question_good_dropout = question_good * dropout_multiplicators
        answer_good_dropout = answer_good * dropout_multiplicators
        question_bad_dropout = question_bad * dropout_multiplicators
        answer_bad_dropout = answer_bad * dropout_multiplicators

        self.similarity_question_answer_good = similarity(
            question_good_dropout,
            answer_good_dropout,
        )
        print(self.similarity_question_answer_good)
        self.similarity_question_answer_good = tf.reshape(self.similarity_question_answer_good, [-1])

        self.similarity_question_answer_bad = similarity(
            question_bad_dropout,
            answer_bad_dropout,
        )
        self.similarity_question_answer_bad = tf.reshape(self.similarity_question_answer_bad, [-1])

        self.loss_individual = hinge_loss(
            self.similarity_question_answer_good,
            self.similarity_question_answer_bad,
            0.3
        )

        reg = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.loss = tf.reduce_mean(self.loss_individual) + reg
        self.predict = self.similarity_question_answer_good

        with tf.name_scope('train_op'):
            vars_list = tf.trainable_variables()
            optimizer = tf.train.AdamOptimizer(self.lr)
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, vars_list), 10)
            self.train_op = optimizer.apply_gradients(zip(grads, vars_list))


def weight_variable(name, shape, regularization=None):
    regularizer = None
    if regularization is not None:
        regularizer = l2_regularizer(1e-5)
    return tf.get_variable(name, shape=shape, initializer=xavier_initializer(), regularizer=regularizer)

def bias_variable(name, shape, value=0.1):
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(value))

def cosine_similarity(a, b):
    return tf.div(
        tf.reduce_sum(tf.multiply(a, b), 1),
        tf.multiply(
            tf.sqrt(tf.reduce_sum(tf.square(a), 1)),
            tf.sqrt(tf.reduce_sum(tf.square(b), 1))
        )
    )

def gesd_similarity(a, b):
    a = tf.nn.l2_normalize(a, dim=1)
    b = tf.nn.l2_normalize(b, dim=1)
    euclidean = tf.sqrt(tf.reduce_sum((a - b) ** 2, 1))
    mm = tf.reshape(
        tf.matmul(
            tf.reshape(a, [-1, 1, tf.shape(a)[1]]),
            tf.transpose(
                tf.reshape(b, [-1, 1, tf.shape(a)[1]]),
                [0, 2, 1]
            )
        ),
        [-1]
    )
    sigmoid_dot = tf.exp(-1 * (mm + 1))
    return 1.0 / (1.0 + euclidean) * 1.0 / (1.0 + sigmoid_dot)

def hinge_loss(similarity_good_tensor, similarity_bad_tensor, margin):
    return tf.maximum(
        0.0,
        tf.add(
            tf.subtract(
                margin,
                similarity_good_tensor
            ),
            similarity_bad_tensor
        )
    )

def weighted_pooling(raw_representation, positional_weighting, tokens):
    """Performs a pooling operation that is similar to average pooling, but uses a weight for every position. Therefore
     not all positions are equally important and contribute equally to the resulting vector.

    :param raw_representation:
    :param positional_weighting:
    :param tokens:
    :return:
    """
    positional_weighting_non_zero = non_zero_tokens(tf.to_float(tokens)) * positional_weighting
    pooled_representation = tf.matmul(
        tf.reshape(positional_weighting_non_zero, [-1, 1, tf.shape(positional_weighting)[1]]),
        raw_representation
    )
    return tf.reshape(pooled_representation, [-1, tf.shape(raw_representation)[2]])

def non_zero_tokens(tokens):
    """Receives a vector of tokens (float) which are zero-padded. Returns a vector of the same size, which has the value
    1.0 in positions with actual tokens and 0.0 in positions with zero-padding.

    :param tokens:
    :return:
    """
    return tf.ceil(tokens / tf.reduce_max(tokens, [1], keep_dims=True))


def soft_alignment(U_AP, b, raw_question_rep, tokens_question_non_zero):
    """Calculate the AP soft-alignment matrix (in a batch-friendly fashion)

    :param U_AP: The AP similarity matrix (to be learned)
    :param raw_question_rep:
    :param raw_answer_rep:
    :param tokens_question_non_zero:
    :param tokens_answer_non_zero:
    :return:
    """
    # Unfortunately, there is no clean way in TF to multiply a 3d tensor with a 2d tensor. We need to perform some
    # reshaping. Compare solution 2 on
    raw_question_rep_flat = tf.reshape(raw_question_rep, [-1, tf.shape(raw_question_rep)[2]])
    QU_flat = tf.matmul(raw_question_rep_flat, U_AP)
    QU = tf.reshape(QU_flat, [-1, tf.shape(raw_question_rep)[1], tf.shape(raw_question_rep)[2]])
    QUA = tf.nn.bias_add(QU, b)
    G = tf.nn.tanh(QUA)

    # We are now removing all the fields of G that belong to zero padding. To achieve this, we are determining these
    # fields and adding a value of -2 to all of them (which is guaranteed to result in a smaller number than the minimum
    # of G, which is -1)
    additions_G_question = tf.transpose(
        tf.reshape((tokens_question_non_zero - 1) * 2, [-1, 1, tf.shape(tokens_question_non_zero)[1]]),
        [0, 2, 1]
    )

    # G_non_zero contains values of less than -1 for all fields which have a relation to zero-padded token positions
    G_non_zero = G + additions_G_question

    return G_non_zero


def attention_softmax(attention, indices_non_zero):
    """Our very own softmax that ignores values of zero token indices (zero-padding)
    :param attention:
    :param raw_indices:
    :return:
    """
    ex = tf.exp(attention) * indices_non_zero
    sum = tf.reduce_sum(ex, [1], keep_dims=True)
    softmax = ex / sum
    return softmax

def attentive_pooling_weights(U_AP, b, raw_answer_rep, tokens_answer, apply_softmax=True):
    """Calculates the attentive pooling weights for question and answer

    :param U_AP: the soft-attention similarity matrix (to learn)
    :param raw_question_rep:
    :param raw_answer_rep:
    :param tokens_question: The raw token indices of the question. Used to detection not-set tokens
    :param tokens_answer: The raw token indices of the answer. Used to detection not-set tokens
    :param Q_PW: Positional weighting matrix for the question
    :param A_PW: Positional weighting matrix for the answer
    :param apply_softmax:
    :return: question weights, answer weights
    """
    tokens_answer_non_zero = non_zero_tokens(tf.to_float(tokens_answer))

    G = soft_alignment(U_AP, b, raw_answer_rep, tokens_answer_non_zero)

    ex = tf.exp(G) * tf.cast(tf.expand_dims(tokens_answer_non_zero, axis=2), tf.float32)
    sum = tf.expand_dims(tf.reduce_sum(ex, axis=2), axis=-1)
    softmax = ex / sum

    attention_answer = softmax * raw_answer_rep

    return attention_answer


