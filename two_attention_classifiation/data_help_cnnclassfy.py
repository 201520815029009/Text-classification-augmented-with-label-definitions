 # -*- coding: utf-8 -*-

import codecs
import pickle
import numpy as np
import random
import json
import re
import jieba
import os
from collections import Counter

PAD_ID = 0
UNK_ID=1
_PAD="_PAD"
_UNK="UNK"

def load_data_multilabel(traning_data_path, valid_data_path, test_data_path, vocab_word2index, sentence_len,
                         sentence_article, article, accusation_label2index, name_scope='cnn'):
    """
    convert data as indexes using word2index dicts.
    :param traning_data_path:
    :param vocab_word2index:
    :param vocab_label2index:
    :return:
    """
    # 1. use cache file if exist
    cache_data_dir = 'cache' + "_" + name_scope
    cache_file =cache_data_dir+"/"+'train_valid_test_article_mnn_word_shiwan_3w.pik'    # train_valid_test_article_32_word
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
    valid_lines = valid_file_object.readlines()
    test_lines = test_data_obejct.readlines()

    random.shuffle(train_lines)
    random.shuffle(valid_lines)
    random.shuffle(test_lines)

    # 3. transform to train/valid data to standardized format
    train = transform_data_to_index(train_lines, vocab_word2index, sentence_len, sentence_article, 'train', article, accusation_label2index)
    valid = transform_data_to_index(valid_lines, vocab_word2index, sentence_len, sentence_article, 'valid', article, accusation_label2index)
    test = transform_data_to_index(test_lines, vocab_word2index, sentence_len, sentence_article, 'test', article, accusation_label2index)

    # 4. save to file system if vocabulary of words not exists
    if not os.path.exists(cache_file):
        with open(cache_file, 'ab') as data_f:
            print("going to dump train/valid/test data to file system.")
            pickle.dump((train, valid, test), data_f, protocol=4)
    return train, valid, test

splitter = ':'
def transform_data_to_index(lines, vocab_word2index, sentence_fact, sentence_article, data_type, article, accusation_label2index):
    """
    transform data to index using vocab and label dict.
    :param lines:
    :param vocab_word2index:
    :param accusation_label2index:
    :param sentence_len: max sentence length
    :return:
    """
    datasample = []
    for i, line in enumerate(lines):
        if i % 10000 == 0:
            print("i:", i)
        json_string = json.loads(line.strip())

        # 1. transform input x.discrete
        facts = json_string['fact']
        input_list = token_string_as_list(facts)  # tokenize
        fact = [vocab_word2index.get(x, UNK_ID) for x in input_list]  # transform input to index
        fact_len = len(fact)
        fact = pad_truncate_list(fact, sentence_fact)

        # 2. transform accusation.discrete
        accusation_list = json_string['meta']['accusation']
        accusation_list = [accusation_label2index[label] for label in accusation_list]
        accusation_list = transform_multilabel_as_multihot(accusation_list, 202)
        ss = []
        for accuname in accusation_label2index.keys():
            input_article_pos = token_string_as_list(accuname + article[accuname])  # tokenize
            article_pos = [vocab_word2index.get(x, UNK_ID) for x in input_article_pos]
            article_pos = pad_truncate_list(article_pos, sentence_article)
            ss.append(article_pos)

        datasample.append((fact, ss, fact_len, 202, accusation_list))

    return datasample

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

lst_stopwords = []
file_stop = open('data/stopwords.txt', 'r', encoding='utf-8')
for line in file_stop.readlines():
    lst_stopwords.append(line.strip('\n'))
stopwords = {}.fromkeys(lst_stopwords)

def token_string_as_list(string,tokenize_style='word'):
    #string=string.decode("utf-8")
    string = replace_money_value(string)
    length = len(string)
    if tokenize_style == 'char':
        listt = [string[i] for i in range(length)]
    elif tokenize_style == 'word':
        listt = jieba.lcut(string)
    listt = [x for x in listt if x.strip()]
    return listt


def replace_money_value(string):
    #print("string:")
    #print(string)
    moeny_list = [1,2,5,7,10, 20, 30,50, 100, 200, 500, 800,1000, 2000, 5000,7000, 10000, 20000, 50000, 80000,100000,200000, 500000, 1000000,3000000,5000000,1000000000]
    double_patten = r'\d+\.\d+'
    #int_patten = r'[\u4e00-\u9fa5,��.����;]\d+[Ԫ����ǧ��ʮ�࣬,��.;��]'
    doubles=re.findall(double_patten,string)
    #ints=re.findall(int_patten,string)
    #ints=[a[1:-1] for a in ints]
    #print(doubles+ints)
    sub_value=0
    for value in (doubles):#+ints):
        for money in moeny_list:
            if money >= float(value):
                sub_value=money
                break
        string=re.sub(str(value),str(sub_value),string)
    return string

def pad_truncate_list(x_list, maxlen):
    """
    :param x_list:e.g. [1,10,3,5,...]
    :return:result_list:a new list,length is maxlen
    """
    result_list = [0 for i in range(maxlen)] #[0,0,..,0]
    length_input = len(x_list)
    if length_input > maxlen: #need to trancat===>no need to pad
        x_list = x_list[0:maxlen]
        for i, element in enumerate(x_list):
            result_list[i] = element
    else:#sequence is to short===>need to pad something===>no need to trancat. [1,2,3], max_len=1000.
        for i in range(length_input):
            result_list[i] = x_list[i]
    return result_list


def create_or_load_vocabulary(predict_path, vocab_size, name_scope='cnn'):
    """
    create vocabulary
    :param vocab_size:
    :param name_scope:
    :return:
    """
    cache_vocabulary_label_pik='cache'+"_"+name_scope # path to save cache
    if not os.path.isdir(cache_vocabulary_label_pik): # create folder if not exists.
        os.makedirs(cache_vocabulary_label_pik)

    #0.if cache exists. load it; otherwise create it.
    cache_path =cache_vocabulary_label_pik+"/"+'vocab_label_article_mn_word_shiwan.pik'  # vocab_label_32_word
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
        file_object = codecs.open('data/data_train.json', mode='r', encoding='utf-8')
        lines = file_object.readlines()
        random.shuffle(lines)

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
        word_vocab_file = predict_path+"/"+'word_freq_shiwan_random.txt'
        if os.path.exists(word_vocab_file):
            print("word vocab file exists.going to delete it.")
            os.remove(word_vocab_file)
        word_freq_file = codecs.open(word_vocab_file,mode='a',encoding='utf-8')
        #put those words to dict
        for i, tuplee in enumerate(vocab_list):
            word,freq = tuplee
            word_freq_file.write(word+":"+str(freq)+"\n")
            vocab_word2index[word] = i+2

        #4.1 accusation and its frequency.
        accusation_freq_file = codecs.open(cache_vocabulary_label_pik+"/"+'accusation_freq_shiwan_random.txt',mode='a',encoding='utf-8')
        accusation_label_list = c_accusation_labels.most_common()
        for i, tuplee in enumerate(accusation_label_list):
            label,freq = tuplee
            accusation_freq_file.write(label+":"+str(freq)+"\n")

        #4.2 accusation dict,  code the accusation with number
        accusation_voc_file = 'data/article-sim-multilabel.txt'
        accusation_voc_object = codecs.open(accusation_voc_file, mode='r', encoding='utf-8')
        accusation_voc_lines = accusation_voc_object.readlines()
        article = {}
        article_len =[]
        for i, accusation_name in enumerate(accusation_voc_lines):
            data = accusation_name.strip('\n').split('\t')
            article_len.append(len(token_string_as_list(accusation_name)))
            article[data[0][:-1]] = data[1]
            accusation_label2index[data[0][:-1]] = i
        article_size = max(article_len)
        #6.save to file system if vocabulary of words not exists.
        if not os.path.exists(cache_path):
            with open(cache_path, 'ab') as data_f:
                print("going to save cache file of vocab of words and labels")
                pickle.dump((vocab_word2index, accusation_label2index, article, article_size), data_f, protocol=4)

    #7.close resources
    word_freq_file.close()
    accusation_freq_file.close()
    print("create_vocabulary.ended")
    return vocab_word2index, accusation_label2index, article, article_size
