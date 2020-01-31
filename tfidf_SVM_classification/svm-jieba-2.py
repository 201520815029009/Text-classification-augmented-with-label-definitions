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
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer

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
accusation_voc_object = codecs.open('data/accu.txt', mode='r', encoding='utf-8')
accusation_voc_lines = accusation_voc_object.readlines()
for i, accusation_name in enumerate(accusation_voc_lines):
    accusation_name = accusation_name.strip()
    accusation_label2index[accusation_name] = i

def get_data(filename):
    # doc_text = ''
    X = []
    Y_accusation = []  # discrete
    data_obejct = codecs.open(filename, mode='r', encoding='utf-8')
    train_lines = data_obejct.readlines()
    random.shuffle(train_lines)
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


train_texts, train_labels = get_data('data/test_high.json')
valid_texts, valid_labels = get_data('data/data_valid.json')
test_texts, test_labels = get_data('data/data_test.json')

#print('************* valid text ********************')
#print(test_texts[0:10])
#print(test_labels[0:10])


all_texts = train_texts + valid_texts + test_texts


'''**************learn the tf-idf of words***************'''
tfidf_vectorizer = TfidfVectorizer(max_df=0.75, min_df=8)#, stop_words=lst_stopwords)
vocab = tfidf_vectorizer.fit(all_texts)
print('************** the length of vocabulary ****************')
print(len(vocab.vocabulary_))
tfidf = vocab.transform(all_texts)

Y = MultiLabelBinarizer().fit_transform(train_labels + valid_labels + test_labels)
print('************ tfidf word frequency matrix **************')
print(tfidf.shape)
print(tfidf[0:10])
print(Y[0:10])


'''************** SVM ***************'''
classif = OneVsRestClassifier(SVC(kernel='linear'))

classif.fit(tfidf[0:len(train_texts)], Y[0:len(train_labels)])
print('*************** the train score is *******************')
#print(classif.score(tfidf[len(train_texts):len(train_texts)+len(valid_texts)], Y[len(train_labels):len(train_labels)+len(valid_labels)]))

#f = open('saved_model/rfc_high.pickle','wb')
#pickle.dump(classif,f)
#f.close()

test_predict = classif.predict(tfidf[len(train_texts)+len(valid_texts):])
print(test_predict[0:10])


label_dict_accusation = init_label_dict(202)
label_dict_accusation = compute_confuse_matrix_batch(test_labels, test_predict, label_dict_accusation)
f1_micro, pre_micro, recal_micro, f1_macro, pre_macro, recal_macro = compute_micro_macro(label_dict_accusation)
print('************************************************')
print("f1_micro_accusation:", f1_micro, "  pre_micro: ", pre_micro, "recal_micro: ", recal_micro)
print("f1_macro_accusation:", f1_macro, "  pre_macro: ", pre_macro, "recal_macro: ", recal_macro)
compute_MAP_MRR(test_predict, test_labels)
