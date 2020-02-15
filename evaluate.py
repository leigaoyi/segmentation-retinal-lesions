# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 18:25:00 2020

@author: kasy
"""



import numpy as np
import cv2

import glob
import os
import json

from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

#import torch
#import torch.nn.functional as F

import re
from tqdm import tqdm

model_name = 'PSPNet'


color_code_labels = [ \
    [0, 0, 0],      # 0 - Black     - Background
    [255, 0, 0],    # 1 - Red       - EX class
    [0, 255, 0],    # 2 - Green     - HE class
    [0, 0, 255],    # 3 - Blue      - MA class
    [255, 255, 0],  # 4 - Yellow    - SE class
    ]               # RGB type



def single_auc(pred_fig, label_fig):
#    pred_ex = 1*(pred_fig[..., 0] == 1)
#    label_ex = 1*(label_fig[..., 0] == 1)
#    
#    #pred_ex_out = pred_ex
#    #label_ex = 1 * (label_fig == color_code_labels[1])
#    #print(pred_ex)
#    pred_ex = np.reshape(pred_ex, [-1])
#    label_ex= np.reshape(label_ex, [-1])
#    label_ex = np.asarray(label_ex, dtype=np.int)
#    #label_ex = np.asarray(label_fig, dtype=np.int)
#    
#    auc = roc_auc_score(label_ex, pred_ex)
#    #print(auc)
    
    
    pred_ex = 1*(pred_fig[..., 0] == 1)
    label_ex = 1*(label_fig[..., 0] == 1)
    
    #pred_ex_out = pred_ex
    #label_ex = 1 * (label_fig == color_code_labels[1])
    #print(pred_ex)
    pred_ex = np.reshape(pred_ex, [-1])
    label_ex= np.reshape(label_ex, [-1])
    label_ex = np.asarray(label_ex, dtype=np.int)
    
    #auc = roc_auc_score(label_fig, pred_fig)
    f1_ex = roc_auc_score(label_ex, pred_ex)
    
    pred_he = 1*(pred_fig[..., 0] == 2)
    label_he = 1*(label_fig[..., 0] == 2)
    
    #pred_ex_out = pred_ex
    #label_ex = 1 * (label_fig == color_code_labels[1])
    #print(pred_ex)
    pred_he = np.reshape(pred_he, [-1])
    label_he= np.reshape(label_he, [-1])
    label_he = np.asarray(label_he, dtype=np.int)
    
    #auc = roc_auc_score(label_fig, pred_fig)
    f1_he = roc_auc_score(label_he, pred_he)
    
    pred_ma = 1*(pred_fig[..., 0] == 3)
    label_ma = 1*(label_fig[..., 0] == 3)
    
    #pred_ex_out = pred_ex
    #label_ex = 1 * (label_fig == color_code_labels[1])
    #print(pred_ex)
    pred_ma = np.reshape(pred_ma, [-1])
    label_ma= np.reshape(label_ma, [-1])
    label_ma = np.asarray(label_ma, dtype=np.int)
    
    #auc = roc_auc_score(label_fig, pred_fig)
    f1_ma = roc_auc_score(label_ma, pred_ma)
    
    pred_se = 1*(pred_fig[..., 0] == 4)
    label_se = 1*(label_fig[..., 0] == 4)
    
    #pred_ex_out = pred_ex
    #label_ex = 1 * (label_fig == color_code_labels[1])
    #print(pred_ex)
    pred_se = np.reshape(pred_se, [-1])
    label_se= np.reshape(label_se, [-1])
    label_se = np.asarray(label_se, dtype=np.int)
    
    #auc = roc_auc_score(label_fig, pred_fig)
    f1_se = roc_auc_score(label_se, pred_se)
    
    auc = [f1_ex, f1_he, f1_ma, f1_se]
    return auc


def single_f1(pred_fig, label_fig):
    pred_ex = 1*(pred_fig[..., 0] == 1)
    label_ex = 1*(label_fig[..., 0] == 1)
    
    #pred_ex_out = pred_ex
    #label_ex = 1 * (label_fig == color_code_labels[1])
    #print(pred_ex)
    pred_ex = np.reshape(pred_ex, [-1])
    label_ex= np.reshape(label_ex, [-1])
    label_ex = np.asarray(label_ex, dtype=np.int)
    
    #auc = roc_auc_score(label_fig, pred_fig)
    f1_ex = f1_score(label_ex, pred_ex)
    
    pred_he = 1*(pred_fig[..., 0] == 2)
    label_he = 1*(label_fig[..., 0] == 2)
    
    #pred_ex_out = pred_ex
    #label_ex = 1 * (label_fig == color_code_labels[1])
    #print(pred_ex)
    pred_he = np.reshape(pred_he, [-1])
    label_he= np.reshape(label_he, [-1])
    label_he = np.asarray(label_he, dtype=np.int)
    
    #auc = roc_auc_score(label_fig, pred_fig)
    f1_he = f1_score(label_he, pred_he)
    
    pred_ma = 1*(pred_fig[..., 0] == 3)
    label_ma = 1*(label_fig[..., 0] == 3)
    
    #pred_ex_out = pred_ex
    #label_ex = 1 * (label_fig == color_code_labels[1])
    #print(pred_ex)
    pred_ma = np.reshape(pred_ma, [-1])
    label_ma= np.reshape(label_ma, [-1])
    label_ma = np.asarray(label_ma, dtype=np.int)
    
    #auc = roc_auc_score(label_fig, pred_fig)
    f1_ma = f1_score(label_ma, pred_ma)
    
    pred_se = 1*(pred_fig[..., 0] == 4)
    label_se = 1*(label_fig[..., 0] == 4)
    
    #pred_ex_out = pred_ex
    #label_ex = 1 * (label_fig == color_code_labels[1])
    #print(pred_ex)
    pred_se = np.reshape(pred_se, [-1])
    label_se= np.reshape(label_se, [-1])
    label_se = np.asarray(label_se, dtype=np.int)
    
    #auc = roc_auc_score(label_fig, pred_fig)
    f1_se = f1_score(label_se, pred_se)
    
    f1 = [f1_ex, f1_he, f1_ma, f1_se]
    #print(auc)
    return f1


#pred_path = './predictions_UNet/IDRiD_13_lab_pred.png'
#label_path = './predictions_UNet/IDRiD_13_EX.tif'


test_num = 67
test_list = [13, 17, 18, 19, 30,\
             33, 47, 50, 52, 56,\
             59, 61, 67, 70, 72, 73]


def print_pred(model_name, test_num):
    pred_path = './predictions_{0}/IDRiD_{1}_lab_pred.png'.format(model_name, test_num)
    label_path = './full_labels/IDRiD_{0}_label.png'.format(test_num)
    
    pred_fig = cv2.imread(pred_path)
    label_fig = cv2.imread(label_path)
    
    pred_fig = cv2.cvtColor(pred_fig, cv2.COLOR_BGR2RGB)
    label_fig = cv2.cvtColor(label_fig, cv2.COLOR_BGR2RGB)
    
    a = single_auc(pred_fig, label_fig)
    #auc = single_auc(pred_fig, label_fig)
    
    print(model_name)
    print('test sampe :', test_num)
    print('auc ex', a[0])
    print('auc he', a[1])
    print('auc ma', a[2])
    print('auc se', a[3])
    
    return (a[0], a[1], a[2], a[3])

#ex_list = []
#he_list = []
#ma_list = []
#se_list = []
#
#for i in test_list:
#    pred_f1 = print_pred(model_name, i)
#    ex_list.append(pred_f1[0])
#    he_list.append(pred_f1[1])
#    ma_list.append(pred_f1[2])
#    se_list.append(pred_f1[3])
#    
#
#
#print(model_name)
#print('EX mean', np.mean(ex_list))
#print('HE mean', np.mean(he_list))
#print('MA mean', np.mean(ma_list))
#print('SE mean', np.mean(se_list))
#
#
#
#with open('./metrics/'+model_name+'_auc.txt', 'w') as f:
#    f.writelines(model_name+'\n')
#    f.writelines('ex '+ str(np.mean(ex_list))+'\n')
#    f.writelines('he '+ str(np.mean(he_list))+'\n')
#    f.writelines('ma '+ str(np.mean(ma_list))+'\n')
#    f.writelines('SE '+ str(np.mean(se_list))+'\n')
    

def dice(y_true, y_pred, smooth = 1):
    true_f = np.reshape(y_true, (-1))
    pred_f = np.reshape(y_pred, (-1))
    
    intersection = np.sum(true_f * pred_f)
    return 2*(intersection + smooth)/(np.sum(true_f)+np.sum(pred_f)+smooth)

def read_test(model_name, test_num):
    pred_path = './predictions_{0}/IDRiD_{1}_lab_pred.png'.format(model_name, test_num)
    label_path = './full_labels/IDRiD_{0}_label.png'.format(test_num)
    
    pred_fig = cv2.imread(pred_path)
    label_fig = cv2.imread(label_path)
    
    pred_fig = cv2.cvtColor(pred_fig, cv2.COLOR_BGR2RGB)[..., 0]
    label_fig = cv2.cvtColor(label_fig, cv2.COLOR_BGR2RGB)[..., 0]
    return pred_fig, label_fig

def compute_EX(label_ex, pred_ex):
    label_ex = 1*(label_ex == 1)
    pred_ex = 1*(pred_ex == 1)
    
    dice_ex = dice(label_ex, pred_ex)
    return dice_ex

def compute_HE(label_ex, pred_ex):
    label_ex = 1*(label_ex == 2)
    pred_ex = 1*(pred_ex == 2)
    
    dice_ex = dice(label_ex, pred_ex)
    return dice_ex

def compute_MA(label_ex, pred_ex):
    label_ex = 1*(label_ex == 3)
    pred_ex = 1*(pred_ex == 3)
    
    dice_ex = dice(label_ex, pred_ex)
    return dice_ex

def compute_SE(label_ex, pred_ex):
    label_ex = 1*(label_ex == 4)
    pred_ex = 1*(pred_ex == 4)
    
    dice_ex = dice(label_ex, pred_ex)
    return dice_ex

#ex_num = test_list[1]
#pred_ex, label_ex = read_test(model_name, ex_num)
#print('Model ', model_name)
#print('EX Dice ', compute_EX(label_ex, pred_ex))
#print('HE Dice ', compute_HE(label_ex, pred_ex))
#print('MA Dice ', compute_MA(label_ex, pred_ex))
#print('SE Dice ', compute_SE(label_ex, pred_ex))
    
EX_list = []
HE_list = []
MA_list = []
SE_list = []

for i in test_list:
    pred_ex, label_ex = read_test(model_name, i)
    print('Model ', model_name, str(i))
    
    a1 = compute_EX(label_ex, pred_ex)
    a2 = compute_HE(label_ex, pred_ex)
    a3 = compute_MA(label_ex, pred_ex)
    a4 = compute_SE(label_ex, pred_ex)
    
    EX_list.append(a1)
    HE_list.append(a2)
    MA_list.append(a3)
    SE_list.append(a4)
    
    print('EX Dice ', a1)
    print('HE Dice ', a2)
    print('MA Dice ', a3)
    print('SE Dice ', a4)
   
print('Model ', model_name)
print('Mean Dice scores :')
print('EX Dice ', np.mean(EX_list))
print('HE Dice ', np.mean(HE_list))
print('MA Dice ', np.mean(MA_list))
print('SE Dice ', np.mean(SE_list))
    
with open('./metrics/'+model_name+'_dice.txt', 'w') as f:
    f.writelines(model_name+'\n')
    f.writelines('ex '+ str(np.mean(EX_list))+'\n')
    f.writelines('he '+ str(np.mean(HE_list))+'\n')
    f.writelines('ma '+ str(np.mean(MA_list))+'\n')
    f.writelines('SE '+ str(np.mean(SE_list))+'\n')    
    
    