# -*- coding: utf-8 -*-
import numpy as np
import random
import operator
"""
compute single evaulation matrix for task1,task2 and task3:
compute f1 score(micro,macro) for accusation & relevant article, and score for pentaly

"""

small_value=0.00001
random_number=2000

def compute_MAP_MRR(y_logits_array, y_targetlabel_list):
    ranks = []
    average_precisions = []
    for k, y_logits_array_single in enumerate(y_logits_array):
        pooled_answers = [k for k in range(0, 202)]
        sorted_answers = sorted(zip(y_logits_array_single, pooled_answers), key=lambda x: -x[0])
        rank = 0
        precisions = []
        for i, (score, answer) in enumerate(sorted_answers, start=1):
            if answer in y_targetlabel_list[k]:
                if rank == 0:
                    rank = i
                precisions.append((len(precisions) + 1) / float(i))
        ranks.append(rank)
        average_precision = np.mean(precisions)
        average_precisions.append(average_precision)
        #print('Rank: {}, AP: {}'.format(rank, average_precision))

    correct_answers = len([a for a in ranks if a == 1])
    accuracy = correct_answers / float(len(ranks))
    mrr = np.mean([1 / float(r) for r in ranks])
    map = np.mean(average_precisions)
    print('Correct answers: {}/{}'.format(correct_answers, len(y_targetlabel_list)))
    print('Accuracy: {}'.format(accuracy))
    print('MRR: {}'.format(mrr))
    print('MAP: {}'.format(map))


def compute_confuse_matrix_batch(y_logits_array, y_targetlabel_list, label_dict, name='default'):
    """
    compute confuse matrix for a batch

    :param y_targetlabel_list: a list; each element is a mulit-hot,e.g. [1,0,0,1,...]
    :param y_logits_array: a 2-d array. [batch_size,num_class]
    :param label_dict:{label:(TP, FP, FN)}
    :param name: a string for debug purpose
    :return:label_dict:{label:(TP, FP, FN)}
    """
    #print('*****************y_logits_array********************')
    #print(y_logits_array[0])
    #print('*****************y_targetlabel_list********************')
    #print(y_targetlabel_list[0])
    acc = 0
    for i, y_logits_array_single in enumerate(y_logits_array):
        label_dict, m = compute_confuse_matrix(y_targetlabel_list[i], y_logits_array_single, label_dict, name=name)
        if m  == True:
            acc = acc+1
    print('#####################  acc  ######################')
    print(acc*1.0/len(y_logits_array ))

    return label_dict

def compute_confuse_matrix(y_targetlabel_list_single, y_logit_array_single, label_dict, name='default'):
    """
    compute true postive(TP), false postive(FP), false negative(FN) given target lable and predict label
    :param y_targetlabel_list: a list. length is batch_size(e.g.1). each element is a multi-hot,like '[0,0,1,0,1,...]'
    :param y_logit_array: an numpy array. shape is:[batch_size,num_classes]
    :param label_dict {label:(TP,FP,FN)}
    :return: macro_f1(a scalar),micro_f1(a scalar)
    """
    #1.get target label and predict label
    # y_target_labels=get_target_label_short(y_targetlabel_list_single) #e.g. y_targetlabel_list[0]=[2,12,88]
    y_target_labels = y_targetlabel_list_single

    # y_predict_labels=[i for i in range(len(y_logit_array_single)) if y_logit_array_single[i]>=0.50] #TODO 0.5PW e.g.[2,12,13,10]
    # y_predict_labels= y_logit_array_single.index(min(y_logit_array_single))

    flag = max(y_logit_array_single)
    y_predict_labels = []
    for i in range(len(y_logit_array_single)):
        if abs(y_logit_array_single[i] - flag) < 0.1:
            y_predict_labels.append(i)

    a = list(set(y_target_labels))
    b = list(set(y_predict_labels))
    acc = operator.eq(a,b)

    #if len(y_predict_labels)<1:    y_predict_labels=[np.argmax(y_logit_array_single)] #TODO ADD 2018.05.29
    if random.choice([x for x in range(random_number)]) ==1:
        print(name+".y_target_labels:",y_target_labels,";y_predict_labels:",y_predict_labels) #debug purpose

    #2.count number of TP,FP,FN for each class
    y_labels_unique=[]
    y_labels_unique.extend(y_target_labels)
    y_labels_unique.extend(y_predict_labels)
    y_labels_unique=list(set(y_labels_unique))
    for i,label in enumerate(y_labels_unique): #e.g. label=2
        TP, FP, FN = label_dict[label]
        if label in y_predict_labels and label in y_target_labels:#predict=1,truth=1 (TP)
            TP=TP+1
        elif label in y_predict_labels and label not in y_target_labels:#predict=1,truth=0(FP)
            FP=FP+1
        elif label not in y_predict_labels and label in y_target_labels:#predict=0,truth=1(FN)
            FN=FN+1
        label_dict[label] = (TP, FP, FN)
    return label_dict, acc


def compute_micro_macro(label_dict):
    """
    compute f1 of micro and macro
    :param label_dict:
    :return: f1_micro,f1_macro: scalar, scalar
    """
    f1_micro, pre_micro, recal_micro = compute_f1_micro_use_TFFPFN(label_dict)
    f1_macro, pre_macro, recal_macro = compute_f1_macro_use_TFFPFN(label_dict)
    return f1_micro, pre_micro, recal_micro, f1_macro, pre_macro, recal_macro


def compute_f1_micro_use_TFFPFN(label_dict):
    """
    compute f1_micro
    :param label_dict: {label:(TP,FP,FN)}
    :return: f1_micro: a scalar
    """
    TF_micro_accusation, FP_micro_accusation, FN_micro_accusation =compute_TF_FP_FN_micro(label_dict)
    f1_micro_accusation, pre, recal = compute_f1(TF_micro_accusation, FP_micro_accusation, FN_micro_accusation,'micro')
    return f1_micro_accusation, pre, recal


def compute_f1_macro_use_TFFPFN(label_dict):
    """
    compute f1_macro
    :param label_dict: {label:(TP,FP,FN)}
    :return: f1_macro
    """
    f1_dict= {}
    pre = 0
    recal = 0
    num_classes=len(label_dict)
    for label, tuplee in label_dict.items():
        TP,FP,FN=tuplee
        f1_score_onelabel, precision, recall =compute_f1(TP,FP,FN,'macro')
        f1_dict[label]=f1_score_onelabel
        pre = pre + precision
        recal = recal + recall
    f1_score_sum=0.0
    for label,f1_score in f1_dict.items():
        f1_score_sum=f1_score_sum+f1_score
    f1_score=f1_score_sum/float(num_classes)
    pre = pre/float(num_classes)
    recal = recal/float(num_classes)
    return f1_score, pre, recal


def compute_f1(TP,FP,FN,compute_type):
    """
    compute f1
    :param TP_micro: number.e.g. 200
    :param FP_micro: number.e.g. 200
    :param FN_micro: number.e.g. 200
    :return: f1_score: a scalar
    """
    precison=TP/(TP+FP+small_value)
    recall=TP/(TP+FN+small_value)
    f1_score=(2*precison*recall)/(precison+recall+small_value)

    if compute_type == 'macro':
        if random.choice([x for x in range(50)]) == 1:
            print(compute_type,"precison:",str(precison),";recall:",str(recall),";f1_score:",f1_score)
    else:
        print(compute_type, "precison:", str(precison), ";recall:", str(recall), ";f1_score:", f1_score)

    return f1_score, precison, recall


def compute_TF_FP_FN_micro(label_dict):
    """
    compute micro FP,FP,FN
    :param label_dict_accusation: a dict. {label:(TP, FP, FN)}
    :return:TP_micro,FP_micro,FN_micro
    """
    TP_micro,FP_micro,FN_micro=0.0,0.0,0.0
    for label,tuplee in label_dict.items():
        TP,FP,FN=tuplee
        TP_micro=TP_micro+TP
        FP_micro=FP_micro+FP
        FN_micro=FN_micro+FN
    return TP_micro,FP_micro,FN_micro

def init_label_dict(num_classes):
    """
    init label dict. this dict will be used to save TP,FP,FN
    :param num_classes:
    :return: label_dict: a dict. {label_index:(0,0,0)}
    """
    label_dict={}
    for i in range(num_classes):
        label_dict[i]=(0,0,0)
    return label_dict

def get_target_label_short(y_mulitihot):
    """
    get target label.
    :param y_mulitihot: [0,0,1,0,1,0,...]
    :return: taget_list.e.g. [3,5,100]
    """
    taget_list = []
    for i, element in enumerate(y_mulitihot):
        if element == 1:
            taget_list.append(i)
    return taget_list