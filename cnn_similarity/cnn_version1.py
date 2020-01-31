
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer, xavier_initializer

tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
class QaCNN(object):
    def __init__(self, trainable_embeddings, question_length, answer_length, embed_size, num_filters, window_size, margin,
                 vocab_size, learning_rate, decay_steps, decay_rate):
        self.trainable_embeddings = trainable_embeddings
        self.question_length = question_length
        self.answer_length = answer_length
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * 0.6)
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.decay_steps, self.decay_rate = decay_steps, decay_rate
        self.margin = margin
        self.num_filters = num_filters
        self.window_size = window_size
        self.similarity = 'cosine'

        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))

        self.input_question = tf.placeholder(tf.int32, [None, self.question_length])
        self.input_answer_good = tf.placeholder(tf.int32, [None, self.answer_length])
        self.input_answer_bad = tf.placeholder(tf.int32, [None, self.answer_length])
        self.dropout_keep_prob = tf.placeholder(tf.float32)

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.embeddings_weight = tf.get_variable("embeddings", [self.vocab_size, self.embed_size], dtype=tf.float32,
                                                initializer=tf.random_normal_initializer(stddev=0.1),
                                                trainable=self.trainable_embeddings)

            self.embeddings_question = tf.nn.embedding_lookup(self.embeddings_weight, self.input_question)
            self.embeddings_answer_good = tf.nn.embedding_lookup(self.embeddings_weight, self.input_answer_good)
            self.embeddings_answer_bad = tf.nn.embedding_lookup(self.embeddings_weight, self.input_answer_bad)

        self.W_conv1 = weight_variable('W_conv', [self.window_size, self.embed_size, 1, self.num_filters])
        self.b_conv1 = bias_variable('b_conv', [self.num_filters])

        raw_representation_question = self.cnn_representation_raw(self.embeddings_question, self.question_length)
        raw_representation_answer_good = self.cnn_representation_raw(self.embeddings_answer_good, self.answer_length)
        raw_representation_answer_bad = self.cnn_representation_raw(self.embeddings_answer_bad, self.answer_length)

        pooled_representation_question = tf.reduce_max(raw_representation_question, [1], keep_dims=False)
        pooled_representation_answer_good = tf.reduce_max(raw_representation_answer_good, [1], keep_dims=False)
        pooled_representation_answer_bad = tf.reduce_max(raw_representation_answer_bad, [1], keep_dims=False)

        self.create_outputs(
            pooled_representation_question,
            pooled_representation_answer_good,
            pooled_representation_question,
            pooled_representation_answer_bad
        )

    def cnn_representation_raw(self, item, sequence_length):
        """Creates a representation graph which retrieves a text item (represented by its word embeddings) and returns
        a vector-representation

        :param item: the text item. Can be question or (good/bad) answer
        :param sequence_length: maximum length of the text item
        :return: representation tensor
        """
        # we need to add another dimension, because cnn works on 3d data only
        cnn_input = tf.expand_dims(item, -1)
        convoluted = tf.nn.bias_add(
            tf.nn.conv2d(
                cnn_input,
                self.W_conv1,
                strides=[1, 1, self.embed_size, 1],
                padding='SAME'
            ),
            self.b_conv1
        )
        return tf.reshape(convoluted, [-1, sequence_length, self.num_filters])

    def maxpool_tanh(item, tokens):
        """Calculates the max-over-time, but ignores the zero-padded positions

        :param item:
        :param tokens:
        :return:
        """
        non_zero = tf.ceil(tf.to_float(tokens) / tf.reduce_max(tokens, [1], keep_dims=True))
        additions = tf.reshape(
            (non_zero - 1) * 2,
            [-1, tf.shape(tokens)[1], 1]
        )
        # additions push the result below -1, which can not be selected in maxpooling
        tanh_item = tf.nn.tanh(item)
        item_processed = tanh_item + additions
        return tf.reduce_max(item_processed, [1], keep_dims=False)

    def create_outputs(self, question_good, answer_good, question_bad, answer_bad):
        similarity_type = self.similarity
        if similarity_type == 'gesd':
            similarity = gesd_similarity
        else:
            similarity = cosine_similarity

        # We apply dropout before similarity. This only works when we dropout the same indices in question and answer.
        # Otherwise, the similarity would be heavily biased (in case of angular/cosine distance).
        dropout_multiplicators = tf.nn.dropout(question_good * 0.0 + 1.0, self.dropout_keep_prob)

        question_good_dropout = question_good * dropout_multiplicators
        answer_good_dropout = answer_good * dropout_multiplicators
        question_bad_dropout = question_bad * dropout_multiplicators
        answer_bad_dropout = answer_bad * dropout_multiplicators

        self.similarity_question_answer_good = similarity(
            question_good_dropout,
            answer_good_dropout,
        )
        self.similarity_question_answer_bad = similarity(
            question_bad_dropout,
            answer_bad_dropout,
        )

        self.loss_individual = hinge_loss(
            self.similarity_question_answer_good,
            self.similarity_question_answer_bad,
            self.margin
        )

        reg = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.loss = tf.reduce_mean(self.loss_individual) + reg
        self.predict = self.similarity_question_answer_good

        with tf.name_scope('train_op'):
            learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                       self.decay_rate, staircase=True)
            self.learning_rate_ = learning_rate
            optimizer = tf.train.AdamOptimizer(learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = optimizer.apply_gradients(zip(gradients, variables))

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
