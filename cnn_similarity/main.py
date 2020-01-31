#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import tensorflow as tf
from gensim.models import KeyedVectors
import numpy as np

from cnn_version1 import QaCNN

from data_helper import create_or_load_vocabulary, load_data_multilabel
from evaluation_matrix import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("data_path","./data","path of traning data.")
tf.app.flags.DEFINE_string("traning_data_file","./data/test_high.json","path of traning data.")
tf.app.flags.DEFINE_string("valid_data_file","./data/data_valid.json","path of validation data.")
tf.app.flags.DEFINE_string("test_data_path","./data/data_test.json","path of validation data.")
tf.app.flags.DEFINE_string("predict_path","./predictor","path of traning data.")
tf.app.flags.DEFINE_string("ckpt_dir","./predictor/checkpoint-word-shiwan-3w-hightext/","checkpoint location for the model") #save to here, so make it easy to upload for test  128-500&113

tf.app.flags.DEFINE_integer("vocab_size",100000,"maximum vocab size.")
tf.app.flags.DEFINE_float("learning_rate",0.0005,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size for training/evaluating.") #32-->128
tf.app.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.") # 32-->128
tf.app.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.") #
tf.app.flags.DEFINE_float("keep_dropout_rate", 0.5, "percentage to keep when using dropout.")
tf.app.flags.DEFINE_integer("sentence_len",500,"max sentence length")
tf.app.flags.DEFINE_integer("sentence_article_len",113,"max sentence length")  #original: 200 word:113   char:215
tf.app.flags.DEFINE_integer("embed_size",64,"embedding size") #64

tf.app.flags.DEFINE_boolean("is_training_flag",True,"is training.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",8,"number of epochs to run.")
tf.app.flags.DEFINE_boolean("use_pretrained_embedding",True,"whether to use embedding or not.")
tf.app.flags.DEFINE_string("word2vec_model_path","data/news_12g_baidubaike_20g_novel_90g_embedding_64.bin","word2vec's vocabulary and vectors") #data_big/law_fasttext_model100.bin-->
#tf.app.flags.DEFINE_string("word2vec_model_path","data_big/law_embedding_64_skipgram.bin","word2vec's vocabulary and vectors")
tf.app.flags.DEFINE_string("name_scope", "cnn", "name scope value.")
tf.app.flags.DEFINE_boolean("multi_label_flag", True, "use multi label or single label.")

def train_cnn():
    # 1.create vocabulary and read data
    vocab_word2index, accusation_label2index, article = create_or_load_vocabulary(FLAGS.predict_path, FLAGS.vocab_size,
                                                                                  name_scope=FLAGS.name_scope)

    vocab_size = len(vocab_word2index)
    print("cnn_model.vocab_size:",vocab_size)
    accusation_num_classes=len(accusation_label2index)
    print("accusation_num_classes:",accusation_num_classes)

    train, valid, valid_answer_list, test, test_answer_list = \
        load_data_multilabel(FLAGS.traning_data_file, FLAGS.valid_data_file, FLAGS.test_data_path, vocab_word2index,
                             FLAGS.sentence_len, FLAGS.sentence_article_len, article, accusation_label2index,
                             name_scope=FLAGS.name_scope)
    print("length of training data:", len(train), ";valid data:", len(valid), ";test data:", len(test))


    #2.create session.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Instantiate Model
        cnn_model = QaCNN(trainable_embeddings=True, question_length=FLAGS.sentence_len, answer_length=FLAGS.sentence_article_len,
                          embed_size=FLAGS.embed_size, num_filters=600, window_size=3, margin=0.3, vocab_size=vocab_size,
                          learning_rate=FLAGS.learning_rate, decay_steps=FLAGS.decay_steps, decay_rate=FLAGS.decay_rate)

        # Initialize Save
        saver = tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint.")
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
            for i in range(2): #decay learning rate if necessary.
                print(i,"Going to decay learning rate by half.")
                sess.run(cnn_model.learning_rate_decay_half_op)
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if FLAGS.use_pretrained_embedding: #load pre-trained word embedding
                vocabulary_index2word={index:word for word,index in vocab_word2index.items()}
                assign_pretrained_word_embedding(sess, vocabulary_index2word, vocab_size, FLAGS.word2vec_model_path,
                                                 cnn_model.embeddings_weight)

        # 3.feed data & training
        curr_epoch=sess.run(cnn_model.epoch_step)
        number_of_training_data=len(train)
        batch_size=FLAGS.batch_size
        accasation_score_best = -100

        #loss, f1_macro_accasation, f1_micro_accasation = do_eval(sess, cnn_model, valid[0:100*202], valid_answer_list[0:100], article)

        for epoch in range(curr_epoch, FLAGS.num_epochs):
            loss_total, counter = 0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size),range(batch_size, number_of_training_data, batch_size)):
                if epoch == 0 and counter == 0:
                    print("trainX[start:end]:",train[start],"train_X.shape:",len(train))
                batch_data = train[start:end]
                q_batch, pos_a_batch, neg_a_batch = zip(*batch_data)

                feed_dict = {cnn_model.input_question: q_batch,
                             cnn_model.input_answer_good: pos_a_batch,
                             cnn_model.input_answer_bad: neg_a_batch,
                             cnn_model.dropout_keep_prob: FLAGS.keep_dropout_rate
                             }

                _, loss = sess.run([cnn_model.train_op, cnn_model.loss], feed_dict)
                loss_total, counter = loss_total + loss, counter + 1

                if counter % 500 == 0:
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f\t" % (epoch, counter, float(loss_total)/float(counter)))

                if start != 0 and start % (20000*FLAGS.batch_size) == 0:
                    # test on dev set
                    loss2, f1_macro_accasation, f1_micro_accasation = do_eval(sess, cnn_model, valid, valid_answer_list, article)
                    accasation_score=((f1_macro_accasation+f1_micro_accasation)/2.0)*100.0
                    print("Epoch %d ValidLoss:%.3f\tMacro_f1_accasation:%.3f\tMicro_f1_accsastion:%.3f\t"%
                          (epoch, loss2, f1_macro_accasation, f1_micro_accasation))
                    print("1.Accasation Score:", accasation_score)
                    # save model to checkpoint
                    if accasation_score > accasation_score_best:
                        save_path = FLAGS.ckpt_dir + "model.ckpt" #TODO temp remove==>only save checkpoint for each epoch once.
                        print("going to save check point.")
                        saver.save(sess, save_path, global_step=epoch)
                        accasation_score_best = accasation_score

            #epoch increment
            print("going to increment epoch counter....")
            sess.run(cnn_model.epoch_increment)

            if (epoch == 1 or epoch == 3 or epoch == 6 or epoch == 9 or epoch == 12 or epoch == 18):
                for i in range(2):
                    print(i, "Going to decay learning rate by half.")
                    sess.run(cnn_model.learning_rate_decay_half_op)

        #print('*******************valid data****************')
        #loss, f1_macro_accasation, f1_micro_accasation = do_eval(sess, cnn_model, valid, valid_answer_list, article)
        #accasation_score = ((f1_macro_accasation + f1_micro_accasation) / 2.0) * 100.0
        #print("ValidLoss:%.3f\tMacro_f1_accasation:%.3f\tMicro_f1_accsastion:%.3f\t" %
        #      (loss, f1_macro_accasation, f1_micro_accasation))
        #print("1.Accasation Score:", accasation_score)

        # 5.Testto 0.0
        _, macrof1, microf1 = do_eval(sess, cnn_model, test, test_answer_list, article)
        f1_all = (macrof1 + microf1) / 2
        print("Test Macro f1:%.3f\tMicro f1:%.3f\t==>f1_all: (Macro_f1 + Micro_f1)/2:%.3f"
              % (macrof1, microf1, f1_all))
        print("testing completed...")


def do_eval(sess, model, valid, valid_answer_list, article_category):
    number_examples = len(valid)
    print("number_examples:", number_examples)
    batch_size = len(article_category)
    print('number of the sample************')
    print(number_examples/batch_size)
    print(len(valid_answer_list))
    q_dev, ans_un = zip(*valid)

    label_dict_accusation = init_label_dict(len(article_category))
    score = []
    loss_total = 0
    counter = 0
    for start, end in zip(range(0, number_examples, batch_size), range(batch_size, number_examples, batch_size)):

        feed_dict = {model.input_question: q_dev[start:end],
                     model.input_answer_good: ans_un[start:end],
                     model.input_answer_bad: ans_un[start:end],
                     model.dropout_keep_prob: 1.0
                     }

        similarity_scores = sess.run(model.similarity_question_answer_good, feed_dict)

        score.append(similarity_scores)
        counter = counter + 1
        #compute confuse matrix for accusation,relevant article,death penalty,life imprisonment
    print('the number of iteration, length of score')
    print(len(score))

    label_dict_accusation = compute_confuse_matrix_batch(score, valid_answer_list, label_dict_accusation, name='accusation')
    #compute f1_micro & f1_macro for accusation,article,deathpenalty,lifeimprisonment
    #f1_micro_accusation,f1_macro_accusation=compute_micro_macro(label_dict_accusation)

    f1_micro, pre_micro, recal_micro, f1_macro, pre_macro, recal_macro = compute_micro_macro(label_dict_accusation)
    print("f1_micro_accusation:", f1_micro, "  pre_micro: ", pre_micro, "  recal_micro: ", recal_micro)
    print("f1_macro_accusation:", f1_macro, "  pre_macro: ", pre_macro, "  recal_macro: ", recal_macro)

    loss2 = loss_total / float(counter)
    print('******* MAP and MRR *************')
    compute_MAP_MRR(score, valid_answer_list)
    return loss2, f1_macro, f1_micro


def assign_pretrained_word_embedding(sess,vocabulary_index2word,vocab_size,word2vec_model_path,embedding_instance):
    print("using pre-trained word emebedding.started.word2vec_model_path:",word2vec_model_path)
    ##word2vec_model = word2vec.load(word2vec_model_path, kind='bin')
    word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True, unicode_errors='ignore')  #
    word2vec_dict = {}
    count_=0
    for word in zip(word2vec_model.vocab): #, word2vec_model.vectors
        if count_==0:
            print("pretrained word embedding size:",str(len(word2vec_model[word])))  #vector
            count_=count_+1
        word2vec_dict[word] = word2vec_model[word] /np.linalg.norm(word2vec_model[word])
    word_embedding_2dlist = [[]] * vocab_size  # create an empty word_embedding list.
    word_embedding_2dlist[0] = np.zeros(FLAGS.embed_size)  # assign empty for first word:'PAD'
    word_embedding_2dlist[1] = np.zeros(FLAGS.embed_size)  # assign empty for first word:'PAD'
    bound = np.sqrt(3.0) / np.sqrt(vocab_size)  # bound for random variables.
    count_exist = 0
    count_not_exist = 0
    for i in range(2, vocab_size):  # loop each word
        word = vocabulary_index2word[i]  # get a word
        embedding = None
        try:
            embedding = word2vec_dict[word]  # try to get vector:it is an array.
        except Exception:
            embedding = None
        if embedding is not None:  # the 'word' exist a embedding
            word_embedding_2dlist[i] = embedding
            count_exist = count_exist + 1  # assign array to this word.
        else:  # no embedding for this word
            word_embedding_2dlist[i] = np.random.uniform(-bound, bound, FLAGS.embed_size)
            count_not_exist = count_not_exist + 1  # init a random value for the word.
    word_embedding_final = np.array(word_embedding_2dlist)  # covert to 2d array.
    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(embedding_instance,word_embedding)  #TODO model.Embedding. assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding)
    print("====>>>>word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")


if __name__ == '__main__':
    train_cnn()