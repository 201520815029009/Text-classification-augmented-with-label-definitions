# -*- coding: utf-8 -*-
#training the model.
#process--->1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')

import tensorflow as tf
import numpy as np
from HAN_model import HierarchicalAttention
from data_util import create_or_load_vocabulary,load_data_multilabel,get_part_validation_data
#data_util_singlelabel

import os
from evaluation_matrix import *
import gensim
from gensim.models import KeyedVectors

#configuration
FLAGS=tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("data_path","./data","path of traning data.")
tf.app.flags.DEFINE_string("traning_data_file","./data/test_high1.json","path of traning data.")
tf.app.flags.DEFINE_string("valid_data_file","./data/data_valid.json","path of validation data.")
tf.app.flags.DEFINE_string("test_data_path","./data/data_test.json","path of validation data.")
tf.app.flags.DEFINE_string("predict_path","./predictor","path of traning data.")
tf.app.flags.DEFINE_string("ckpt_dir","./predictor/checkpoint-shiwan-han_high/","checkpoint location for the model") #save to here, so make it easy to upload for test

tf.app.flags.DEFINE_integer("vocab_size",100000,"maximum vocab size.")
tf.app.flags.DEFINE_float("learning_rate",0.001,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size for training/evaluating.") #批处理的大小 32-->128
tf.app.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.") #6000批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.") #0.65一次衰减多少
tf.app.flags.DEFINE_float("keep_dropout_rate", 0.5, "percentage to keep when using dropout.") #0.65一次衰减多少
tf.app.flags.DEFINE_integer("sentence_len",500,"max sentence length")
tf.app.flags.DEFINE_integer("num_sentences",10,"number of sentences")
tf.app.flags.DEFINE_integer("embed_size",64,"embedding size") #64
tf.app.flags.DEFINE_integer("hidden_size",128,"hidden size") #128
tf.app.flags.DEFINE_integer("num_filters",128,"number of filter for a filter map used in CNN.") #128

tf.app.flags.DEFINE_boolean("is_training_flag",True,"is training.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",20,"number of epochs to run.")  #20
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.") #每10轮做一次验证
tf.app.flags.DEFINE_boolean("use_pretrained_embedding",True,"whether to use embedding or not.")
tf.app.flags.DEFINE_string("word2vec_model_path","data/news_12g_baidubaike_20g_novel_90g_embedding_64.bin","word2vec's vocabulary and vectors") #data_big/law_fasttext_model100.bin-->
#tf.app.flags.DEFINE_string("word2vec_model_path","data_big/law_embedding_64_skipgram.bin","word2vec's vocabulary and vectors")
tf.app.flags.DEFINE_string("name_scope", "cnn", "name scope value.")
tf.app.flags.DEFINE_boolean("multi_label_flag", True, "use multi label or single label.")
tf.app.flags.DEFINE_boolean("test_mode", False, "whether it is test mode. if it is test mode, only small percentage of data will be used")

tf.app.flags.DEFINE_string("model", "han", "name of model:han,text_cnn,dp_cnn,c_gru,c_gru2,gru,pooling")  #text_cnn
tf.app.flags.DEFINE_string("pooling_strategy", "hier","pooling strategy used when model is pooling. {avg,max,concat,hier}")
#you can change this
filter_sizes=[2,3,4,5,6,7,8]# [6, 7, 8, 9, 10]

stride_length=1
def main(_):
    print("model:",FLAGS.model)
    vocab_word2index, accusation_label2index = create_or_load_vocabulary(FLAGS.data_path, FLAGS.predict_path,
                                                                         FLAGS.traning_data_file, FLAGS.vocab_size,
                                                                         name_scope=FLAGS.name_scope, test_mode=FLAGS.test_mode)

    vocab_size = len(vocab_word2index)
    print("cnn_model.vocab_size:",vocab_size)
    accusation_num_classes=len(accusation_label2index)
    print("accusation_num_classes:",accusation_num_classes)

    train, valid, test = load_data_multilabel(FLAGS.traning_data_file, FLAGS.valid_data_file, FLAGS.test_data_path,
                                              vocab_word2index, accusation_label2index, FLAGS.sentence_len,
                                              name_scope=FLAGS.name_scope, test_mode=FLAGS.test_mode)
    train_X, train_Y_accusation = train
    valid_X, valid_Y_accusation = valid
    test_X, test_Y_accusation = test
    #print some message for debug purpose
    print("length of training data:",len(train_X),";valid data:",len(valid_X),";test data:",len(test_X))
    print("trainX_[0]:", train_X[0])
    train_Y_accusation_short = get_target_label_short(train_Y_accusation[0])
    print("train_Y_accusation_short:", train_Y_accusation_short)

    #2.create session.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        #Instantiate Model
        model=HierarchicalAttention(accusation_num_classes, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps,
                                    FLAGS.decay_rate, FLAGS.sentence_len, FLAGS.num_sentences, vocab_size,
                                    FLAGS.embed_size, FLAGS.hidden_size, num_filters=FLAGS.num_filters, model=FLAGS.model,
                                    filter_sizes=filter_sizes,stride_length=stride_length,pooling_strategy=FLAGS.pooling_strategy)
        #Initialize Save
        saver = tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint.")
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
            for i in range(1): #decay learning rate if necessary.
                print(i,"Going to decay learning rate by half.")
                sess.run(model.learning_rate_decay_half_op)
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if FLAGS.use_pretrained_embedding: #load pre-trained word embedding
                vocabulary_index2word={index:word for word,index in vocab_word2index.items()}
                assign_pretrained_word_embedding(sess, vocabulary_index2word, vocab_size, model,FLAGS.word2vec_model_path,model.Embedding)

        curr_epoch=sess.run(model.epoch_step)
        #3.feed data & training
        number_of_training_data=len(train_X)
        batch_size=FLAGS.batch_size
        iteration=0
        accasation_score_best=-100

        for epoch in range(curr_epoch, FLAGS.num_epochs):
            loss_total, counter = 0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size),range(batch_size, number_of_training_data, batch_size)):
                iteration = iteration+1
                if epoch == 0 and counter == 0:
                    print("trainX[start:end]:",train_X[start:end],"train_X.shape:",train_X.shape)
                feed_dict = {model.input_x: train_X[start:end], model.input_y_accusation: train_Y_accusation[start:end],
                             model.dropout_keep_prob: FLAGS.keep_dropout_rate, model.is_training_flag: FLAGS.is_training_flag}
                             #model.iter: iteration, model.tst: not FLAGS.is_training
                current_loss,lr,loss_accusation, l2_loss,_= sess.run([model.loss_val,model.learning_rate,
                                                                      model.loss_accusation,model.l2_loss,
                                                                      model.train_op],feed_dict)
                loss_total, counter = loss_total + current_loss, counter + 1
                if counter % 100 == 0:
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tLearning rate:%.5f" %(epoch,counter,float(loss_total)/float(counter),lr))
                if counter % 200 == 0:
                    print("Loss_accusation:%.3f\tL2_loss:%.3f\tCurrent_loss:%.3f\t"
                          % (loss_accusation, l2_loss, current_loss))
                ########################################################################################################
                if start != 0 and start % (2000*FLAGS.batch_size) == 0: # eval every 400 steps.
                    loss, f1_macro_accasation, f1_micro_accasation = \
                        do_eval(sess, model, valid, iteration, accusation_num_classes)
                    accasation_score=((f1_macro_accasation+f1_micro_accasation)/2.0)*100.0
                    score_all=accasation_score #3ecfDzJbjUvZPUdS
                    print("Epoch %d ValidLoss:%.3f\tMacro_f1_accasation:%.3f\tMicro_f1_accsastion:%.3f\t"
                                % (epoch, loss, f1_macro_accasation, f1_micro_accasation))
                    print("1.Accasation Score:", accasation_score, ";Score ALL:", score_all)
                    # save model to checkpoint
                    if accasation_score>accasation_score_best:
                        save_path = FLAGS.ckpt_dir + "model.ckpt" #TODO temp remove==>only save checkpoint for each epoch once.
                        print("going to save check point.")
                        saver.save(sess, save_path, global_step=epoch)
                        accasation_score_best=accasation_score
            #epoch increment
            print("going to increment epoch counter....")
            sess.run(model.epoch_increment)

            # 4.validation
            print(epoch,FLAGS.validate_every,(epoch % FLAGS.validate_every==0))
            if epoch % FLAGS.validate_every==0:
                loss, f1_macro_accasation, f1_micro_accasation = \
                    do_eval(sess,model,valid,iteration,accusation_num_classes)
                accasation_score = ((f1_macro_accasation + f1_micro_accasation) / 2.0) * 100.0

                print()
                print("Epoch %d ValidLoss:%.3f\tMacro_f1_accasation:%.3f\tMicro_f1_accsastion:%.3f\t"
                      % (epoch, loss, f1_macro_accasation, f1_micro_accasation))
                print("===>1.Accasation Score ( (Macro_f1 + Micro_f1)/2 ):", accasation_score)

                #save model to checkpoint
                if accasation_score > accasation_score_best:
                    save_path = FLAGS.ckpt_dir+"model.ckpt"
                    print("going to save check point.")
                    saver.save(sess, save_path, global_step=epoch)
                    accasation_score_best = accasation_score
            #if (epoch == 2 or epoch == 4 or epoch == 7 or epoch==10 or epoch == 13  or epoch==19):
            if (epoch == 1 or epoch == 3 or epoch == 5 or epoch == 7 or epoch == 12 or epoch == 18):
                for i in range(2):
                    print(i, "Going to decay learning rate by half.")
                    sess.run(model.learning_rate_decay_half_op)


        # 5.最后在测试集上做测试，并报告测试准确率 Testto 0.0
        test_loss, macrof1, microf1 = do_eval(sess, model, test, iteration, accusation_num_classes)
        mac_min_f1 = (macrof1+microf1)/2
        print("Test Loss:%.3f\tMacro f1:%.3f\tMicro f1:%.3f\t==>(Macro_f1 + Micro_f1)/2:%.3f"
              % (test_loss, macrof1, microf1, mac_min_f1))
        print("training completed...")
    pass

def do_eval(sess,model,valid,iteration,accusation_num_classes):
    valid_X, valid_Y_accusation = get_part_validation_data(valid)
    number_examples = len(valid_X)
    print("number_examples:", number_examples)
    batch_size = FLAGS.batch_size
    label_dict_accusation = init_label_dict(accusation_num_classes)

    score = []
    loss_total = 0
    counter = 0

    for start, end in zip(range(0, number_examples, batch_size), range(batch_size, number_examples, batch_size)):
        feed_dict = {model.input_x: valid_X[start:end],
                     model.input_y_accusation: valid_Y_accusation[start:end],
                     model.dropout_keep_prob: 1.0, model.is_training_flag: False}

        loss, similarity_scores = sess.run([model.loss_val, model.logits_accusation], feed_dict)
        if start == 0:
            print(similarity_scores)
        score.extend(similarity_scores)
        counter = counter + 1
        loss_total= loss_total+ loss

    label_dict_accusation = compute_confuse_matrix_batch(valid_Y_accusation, score, label_dict_accusation, name='accusation')
    #compute f1_micro & f1_macro for accusation,article,deathpenalty,lifeimprisonment
    #f1_micro_accusation,f1_macro_accusation=compute_micro_macro(label_dict_accusation)

    f1_micro, pre_micro, recal_micro, f1_macro, pre_macro, recal_macro = compute_micro_macro(label_dict_accusation)
    print("f1_micro_accusation:", f1_micro, "  pre_micro: ", pre_micro, "  recal_micro: ", recal_micro)
    print("f1_macro_accusation:", f1_macro, "  pre_macro: ", pre_macro, "  recal_macro: ", recal_macro)

    loss2 = loss_total / float(counter)
    print('******* MAP and MRR *************')
    compute_MAP_MRR(score, valid_Y_accusation)
    return loss2, f1_macro, f1_micro

    return eval_loss/float(eval_counter+small_value), f1_macro_accusation, f1_micro_accusation


def assign_pretrained_word_embedding(sess,vocabulary_index2word,vocab_size,model,word2vec_model_path,embedding_instance):
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
    count_exist = 0;
    count_not_exist = 0
    for i in range(2, vocab_size):  # loop each word
        word = vocabulary_index2word[i]  # get a word
        embedding = None
        try:
            embedding = word2vec_dict[word]  # try to get vector:it is an array.
        except Exception:
            embedding = None
        if embedding is not None:  # the 'word' exist a embedding
            word_embedding_2dlist[i] = embedding;
            count_exist = count_exist + 1  # assign array to this word.
        else:  # no embedding for this word
            word_embedding_2dlist[i] = np.random.uniform(-bound, bound, FLAGS.embed_size);
            count_not_exist = count_not_exist + 1  # init a random value for the word.
    word_embedding_final = np.array(word_embedding_2dlist)  # covert to 2d array.
    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(embedding_instance,word_embedding)  #TODO model.Embedding. assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding);
    print("====>>>>word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")


if __name__ == "__main__":
    tf.app.run()