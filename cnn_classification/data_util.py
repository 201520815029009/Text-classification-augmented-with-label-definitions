# -*- coding: utf-8 -*-
#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8') #gb2312
import codecs
import random
import numpy as np
from tflearn.data_utils import pad_sequences
from collections import Counter
import os
import pickle
import json
import jieba
from predictor.data_util_test import pad_truncate_list

PAD_ID = 0
UNK_ID=1
_PAD="_PAD"
_UNK="UNK"


def load_data_multilabel(traning_data_path, valid_data_path, test_data_path, vocab_word2index, accusation_label2index,
                         sentence_len, name_scope='cnn', test_mode=False):
    """
    convert data as indexes using word2index dicts.
    :param traning_data_path:
    :param vocab_word2index:
    :param vocab_label2index:
    :return:
    """
    # 1. use cache file if exist
    cache_data_dir = 'cache' + "_" + name_scope
    cache_file =cache_data_dir+"/"+'train_valid_test_shiwan_3w_high.pik'
    print("cache_path:",cache_file,"train_valid_test_file_exists:",os.path.exists(cache_file))
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as data_f:
            print("going to load cache file from file system and return")
            return pickle.load(data_f)
    # 2. read source file
    train_file_object = codecs.open(traning_data_path, mode='r', encoding='utf-8')
    valid_file_object = codecs.open(valid_data_path, mode='r', encoding='utf-8')
    test_data_obejct = codecs.open(test_data_path, mode='r', encoding='utf-8')
    train_lines = train_file_object.readlines()
    valid_lines=valid_file_object.readlines()
    test_lines=test_data_obejct.readlines()

    random.shuffle(train_lines)
    random.shuffle(valid_lines)
    random.shuffle(test_lines)

    if test_mode:
        train_lines=train_lines[0:1000]
    # 3. transform to train/valid data to standardized format
    train = transform_data_to_index(train_lines, vocab_word2index, accusation_label2index, sentence_len,'train',name_scope)
    valid = transform_data_to_index(valid_lines, vocab_word2index, accusation_label2index, sentence_len,'valid',name_scope)
    test = transform_data_to_index(test_lines, vocab_word2index, accusation_label2index, sentence_len,'test',name_scope)

    # 4. save to file system if vocabulary of words not exists
    if not os.path.exists(cache_file):
        with open(cache_file, 'ab') as data_f:
            print("going to dump train/valid/test data to file system.")
            pickle.dump((train,valid,test),data_f, protocol=4)
    return train, valid, test

splitter = ':'
num_mini_examples=1900
def transform_data_to_index(lines, vocab_word2index, accusation_label2index, sentence_len, data_type, name_scope):
    """
    transform data to index using vocab and label dict.
    :param lines:
    :param vocab_word2index:
    :param accusation_label2index:
    :param article_label2index:
    :param deathpenalty_label2index:
    :param lifeimprisonment_label2index:
    :param sentence_len: max sentence length
    :return:
    """
    X = []
    Y_accusation = []  # discrete
    accusation_label_size=len(accusation_label2index)

    # load frequency of accu and relevant articles, so that we can copy those data with label are few. ADD 2018-05-29
    accusation_freq_dict = load_accusation_freq_dict(accusation_label2index, name_scope)

    for i, line in enumerate(lines):
        if i % 10000 == 0:
            print("i:", i)
        json_string = json.loads(line.strip())

        # 1. transform input x.discrete
        facts = json_string['fact']
        input_list = token_string_as_list(facts)  # tokenize
        x = [vocab_word2index.get(x, UNK_ID) for x in input_list]  # transform input to index
        x = pad_truncate_list(x, sentence_len)

        # 2. transform accusation.discrete
        accusation_list = json_string['meta']['accusation']
        accusation_list = [accusation_label2index[label] for label in accusation_list]
        y_accusation = transform_multilabel_as_multihot(accusation_list, accusation_label_size)

        # OVER-SAMPLING:if it is training data, copy labels that are few based on their frequencies.
        # num_copy = 1
        # if data_type == 'train': #set specially weight and copy some examples when it is training data.
        #     freq_accusation = accusation_freq_dict[accusation_list[0]]
        #     if freq_accusation <= num_mini_examples:
        #         freq = freq_accusation
        #         num_copy=max(1,num_mini_examples/freq)
        #         if i%1000==0:
        #             print("####################freq_accusation:", freq_accusation, ";num_copy:", num_copy)
        #
        # for k in range(int(num_copy)):
        #     X.append(x)
        #     Y_accusation.append(y_accusation)

        #### no oversampling
        X.append(x)
        Y_accusation.append(y_accusation)
    #shuffle
    number_examples = len(X)
    X_ = []
    Y_accusation_ = []
    permutation = np.random.permutation(number_examples)
    for index in permutation:
        X_.append(X[index])
        Y_accusation_.append(Y_accusation[index])
    X_ = np.array(X_)
    data = (X_, Y_accusation_)
    return data

def transform_multilabel_as_multihot(label_list,label_size):
    """
    convert to multi-hot style
    :param label_list: e.g.[0,1,4], here 4 means in the 4th position it is true value(as indicate by'1')
    :param label_size: e.g.199
    :return:e.g.[1,1,0,1,0,0,........]
    """
    result=np.zeros(label_size)
    #set those location as 1, all else place as 0.
    result[label_list] = 1
    return result

def transform_mulitihot_as_dense_list(multihot_list):
    length = len(multihot_list)
    result_list = [i for i in range(length) if multihot_list[i] > 0]
    return result_list

#use pretrained word embedding to get word vocabulary and labels, and its relationship with index
def create_or_load_vocabulary(data_path, predict_path, training_data_path, vocab_size, name_scope='cnn', test_mode=False):
    """
    create vocabulary
    :param training_data_path:
    :param vocab_size:
    :param name_scope:
    :return:
    """
    cache_vocabulary_label_pik='cache'+"_"+name_scope # path to save cache
    if not os.path.isdir(cache_vocabulary_label_pik): # create folder if not exists.
        os.makedirs(cache_vocabulary_label_pik)

    #0.if cache exists. load it; otherwise create it.
    cache_path =cache_vocabulary_label_pik+"/"+'vocab_label_shiwan_high.pik'
    print("cache_path:",cache_path,"file_exists:",os.path.exists(cache_path))
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as data_f:
            print("going to load cache file.vocab of words and labels")
            return pickle.load(data_f)
    else:
        vocab_word2index = {}
        vocab_word2index[_PAD] = PAD_ID
        vocab_word2index[_UNK] = UNK_ID

        accusation_label2index = {}

        #1.load raw data
        file_object = codecs.open(training_data_path, mode='r', encoding='utf-8')
        lines = file_object.readlines()
        random.shuffle(lines)
        if test_mode:
           lines=lines[0:10000]

        #2.loop each line,put to counter
        c_inputs = Counter()
        c_accusation_labels = Counter()
        for i,line in enumerate(lines):
            if i % 10000 == 0:
                print(i)
            json_string = json.loads(line.strip())
            facts = json_string['fact']
            input_list = token_string_as_list(facts)
            c_inputs.update(input_list)

            accusation_list = json_string['meta']['accusation']
            c_accusation_labels.update(accusation_list)

        #3.get most frequency words
        vocab_list = c_inputs.most_common(vocab_size)
        word_vocab_file = predict_path+"/"+'word_freq_shiwan_3w_high.txt'
        if os.path.exists(word_vocab_file):
            print("word vocab file exists.going to delete it.")
            os.remove(word_vocab_file)
        word_freq_file = codecs.open(word_vocab_file,mode='a',encoding='utf-8')
        for i, tuplee in enumerate(vocab_list):
            word,freq = tuplee
            word_freq_file.write(word+":"+str(freq)+"\n")
            vocab_word2index[word] = i+2

        #4.1 accusation and its frequency.
        accusation_freq_file = codecs.open(cache_vocabulary_label_pik+"/"+'accusation_freq_shiwan_3w_high.txt',mode='a',encoding='utf-8')
        accusation_label_list = c_accusation_labels.most_common()
        for i, tuplee in enumerate(accusation_label_list):
            label,freq = tuplee
            accusation_freq_file.write(label+":"+str(freq)+"\n")


        #4.2 accusation dict,  code the accusation with number
        accusation_voc_file = data_path+"/accu.txt"
        accusation_voc_object = codecs.open(accusation_voc_file,mode='r',encoding='utf-8')
        accusation_voc_lines = accusation_voc_object.readlines()
        for i, accusation_name in enumerate(accusation_voc_lines):
            accusation_name=accusation_name.strip()
            accusation_label2index[accusation_name] = i

        #6.save to file system if vocabulary of words not exists.
        if not os.path.exists(cache_path):
            with open(cache_path, 'ab') as data_f:
                print("going to save cache file of vocab of words and labels")
                pickle.dump((vocab_word2index, accusation_label2index), data_f, protocol=4)

    #7.close resources
    word_freq_file.close()
    accusation_freq_file.close()
    print("create_vocabulary.ended")
    return vocab_word2index, accusation_label2index

def token_string_as_list(string,tokenize_style='word'):
    #string=string.decode("utf-8")
    string = replace_money_value(string)  #TODO add normalize number ADD 2018.06.11
    length = len(string)
    if tokenize_style == 'char':
        listt = [string[i] for i in range(length)]
    elif tokenize_style == 'word':
        listt = jieba.lcut(string)
    listt = [x for x in listt if x.strip()]
    return listt

def get_part_validation_data(valid, num_valid=6000):
    valid_X, valid_Y_accusation = valid
    number_examples = len(valid_X)
    permutation = np.random.permutation(number_examples)[0:num_valid]
    valid_X2, valid_Y_accusation2 = [], []
    for index in permutation:
        valid_X2.append(valid_X[index])
        valid_Y_accusation2.append(valid_Y_accusation[index])
    return valid_X2, valid_Y_accusation2


def load_accusation_freq_dict(accusation_label2index, name_scope):
    cache_vocabulary_label_pik = 'cache'+"_"+name_scope # path to save cache
    #load dict of accusations
    accusation_freq_file = codecs.open(cache_vocabulary_label_pik + "/" + 'accusation_freq_shiwan_3w_high.txt', mode='r',encoding='utf-8')
    accusation_freq_lines = accusation_freq_file.readlines()
    accusation_freq_dict = {}
    for i, line in enumerate(accusation_freq_lines):
        acc_label, freq = line.strip().split(splitter) #编造、故意传播虚假恐怖信息:122
        accusation_freq_dict[accusation_label2index[acc_label]] = int(freq)
    return accusation_freq_dict

import re
def replace_money_value(string):
    #print("string:")
    #print(string)
    moeny_list = [1,2,5,7,10, 20, 30,50, 100, 200, 500, 800,1000, 2000, 5000,7000, 10000, 20000, 50000, 80000,100000,200000, 500000, 1000000,3000000,5000000,1000000000]
    double_patten = r'\d+\.\d+'
    int_patten = r'[\u4e00-\u9fa5,，.。；;]\d+[元块万千百十余，,。.;；]'
    doubles=re.findall(double_patten,string)
    ints=re.findall(int_patten,string)
    ints=[a[1:-1] for a in ints]
    #print(doubles+ints)
    sub_value=0
    for value in (doubles+ints):
        for money in moeny_list:
            if money >= float(value):
                sub_value=money
                break
        string=re.sub(str(value),str(sub_value),string)
    return string

