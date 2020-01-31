#!/usr/bin/python
# -*- coding: utf-8 -*-
import codecs
import json
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import metrics
from evaluation_matrix import *

import datetime

print(datetime.datetime.now())
'''**************cut the document and remove the stopwords***************'''
lst_stopwords = []
file_stop = open('data/stopwords.txt', 'r', encoding='utf-8')
for line in file_stop.readlines():
    lst_stopwords.append(line.strip('\n'))
stopwords = {}.fromkeys(lst_stopwords)

def preprocessing_doc(raw_doc):
    word_lst = jieba.cut(raw_doc) #raw_doc.split()
    stp_free_text = " ".join([word for word in word_lst if word not in stopwords])
    return stp_free_text

'''**************read the data***************'''
accusation_label2index = {}
accusation_voc_object = codecs.open('data-multi-label/article-sim-multilable.txt', mode='r', encoding='utf-8')
accusation_voc_lines = accusation_voc_object.readlines()
article = []
category = []
for i, accusation_name in enumerate(accusation_voc_lines):
    data = accusation_name.strip('\n').split('\t')
    article.append(preprocessing_doc(data[1]))
    category.append(i)
    accusation_label2index[data[0][:-1]] = i
print(accusation_label2index)

def get_data(filename):
    # doc_text = ''
    X = []
    Y_accusation = []  # discrete
    data_obejct = codecs.open(filename, mode='r', encoding='utf-8')
    train_lines = data_obejct.readlines()
    for i, line in enumerate(train_lines):
        if i % 10000 == 0:
            print("i:", i)
        json_string = json.loads(line.strip())

        # 1. transform input x.discrete
        facts = json_string['fact']
        facts = preprocessing_doc(facts)
        X.append(facts)
        # 2. transform accusation.discrete
        accusation_list = json_string['meta']['accusation']
        accusation_list = [accusation_label2index[label] for label in accusation_list]
        Y_accusation.append(accusation_list)
    return X, Y_accusation

train_texts, train_labels = get_data('data-multi-label/test_high.json')  #data_train
valid_texts, valid_labels = get_data('data-multi-label/data_valid.json')
test_texts, test_labels = get_data('data-multi-label/data_test.json')

print('***********************test data ****************************')
print(test_labels[0])
print(test_texts[0])
print('length of the article******************************')
print(len(article))
print(article[0])

all_texts = article
all_texts.extend(test_texts)
all_texts.extend(train_texts)
all_texts.extend(valid_texts)

'''**************learn the tf-idf of words***************'''
tfidf_vectorizer = TfidfVectorizer(max_df=0.75, min_df=8)#, stop_words=lst_stopwords)   # 0.85  5
#该类会统计每个词语的tf-idf权值
transformer = TfidfTransformer()
#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
tfidf = transformer.fit_transform(tfidf_vectorizer.fit_transform(all_texts))

#获取词袋模型中的所有词语
# word = tfidf_vectorizer.get_feature_names()
# weight = tfidf.toarray()
#
# f = open("data-multi-label/TIAsmilarity.txt", "w")
# #打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for遍历某一类文本下的词语权重
# for i in range(len(weight)):
# # print u"-------这里输出第",i,u"类文本的词语tf-idf权重------"
#     f.write(str(i+1)+"\t")
#     for j in range(len(word)):
#         if weight[i][j] > 0:
#             f.write(str(j+1) + ":" + str(weight[i][j]) + " ")
#     f.write("\n")
#     print(i)
# f.close()

print('**************************************')
print(len(test_labels))

'''**************compute the similarity***************'''
SimMatrix = (tfidf[0:202+len(test_labels)] * tfidf[0:202+len(test_labels)].T).A 
print(SimMatrix[202][107]) #"第一篇与第4篇的相似度"

y_pred_cls = []
flag = 0
for i in range(len(category), len(category)+len(test_labels)):
    pre_sub = []
    for j in range(0, len(category)):
        pre_sub.append(SimMatrix[i][j])
    y_pred_cls.append(pre_sub)
    flag = i
print('***************************************')
print(flag)
print(y_pred_cls[0])

label_dict_accusation = init_label_dict(202)
label_dict_accusation = compute_confuse_matrix_batch(test_labels, y_pred_cls, label_dict_accusation)
f1_micro, pre_micro, recal_micro, f1_macro, pre_macro, recal_macro = compute_micro_macro(label_dict_accusation)
print('************************************************')
print("f1_micro_accusation:", f1_micro, "  pre_micro: ", pre_micro, "recal_micro: ", recal_micro)
print("f1_macro_accusation:", f1_macro, "  pre_macro: ", pre_macro, "recal_macro: ", recal_macro)
compute_MAP_MRR(y_pred_cls, test_labels)
print(datetime.datetime.now())

