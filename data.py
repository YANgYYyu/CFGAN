# -*- coding: utf-8 -*-

"""
Created on Mar. 11, 2019.
tensorflow implementation of the paper:
Dong-Kyu Chae et al. "CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks," In Proc. of ACM CIKM, 2018.
@author: Dong-Kyu Chae (kyu899@agape.hanyang.ac.kr)

IMPORTANT: make sure that (1) the user & item indices start from 0, and (2) the index should be continuous, without any empy index.
"""

import random
from collections import defaultdict
import torch


def split_data():
    random.seed(0)
    print("start")
    fp1 = open("data/ml_small/train.csv", mode="w")
    fp2 = open("data/ml_small/test.csv", mode="w")
    for line in open("data/ml_small/ratings.csv"):
        if random.randint(0, 8) == 0:
            fp2.writelines(line)
        else:
            fp1.writelines(line)
    print("end")
    fp1.close()
    fp2.close()


'''加载训练集
trainFile  str 训练集文件名
splitMark  str 文件一行中的分隔符
'''


def load_training_data(train_file, split_mark):
    # trainFile = path + "/" + benchmark + "/" + benchmark + ".train"
    print(train_file)

    train_set = defaultdict(list)  # 字典默认值是一个list    train_set['key_new'] 是个list
    max_u_id = -1
    max_i_id = -1

    for line in open(train_file):
        user_id, item_id, rating, _ = line.strip().split(split_mark)

        user_id = int(user_id)
        item_id = int(item_id)

        # note that we regard all the observed ratings as implicit feedback
        train_set[user_id].append(item_id)

        max_u_id = max(user_id, max_u_id)
        max_i_id = max(item_id, max_i_id)

    for u, i_list in train_set.items():
        i_list.sort()

    user_count = max_u_id + 1
    item_count = max_i_id + 1

    print(user_count)
    print(item_count)

    print("Training data loading done: %d users, %d items" % (user_count, item_count))

    return train_set, user_count, item_count  # 此处userCount itemCount并不能代表真实值  因为可能小于测试集合中的userCount item_count


'''
装载测试集数据
testSet [1:[...],2[...], ...]  defaultdict(list)
GroundTruth   [[],[], ...] 只装着item
'''


def load_test_data(test_file, split_mark):
    test_set = defaultdict(list)
    for line in open(test_file):
        user_id, item_id, rating, _ = line.strip().split(split_mark)
        user_id = int(user_id)
        item_id = int(item_id)

        # note that we regard all the ratings in the test set as ground truth
        test_set[user_id].append(item_id)

    ground_truth = []
    for u, i_list in test_set.items():
        tmp = []
        for j in i_list:
            tmp.append(j)

        ground_truth.append(tmp)

    print("Test data loading done")

    return test_set, ground_truth


''' 返回量trainVector test_mask_vector batchCount
testMaskVector 与trainVector相对应  -99999对应1
testMaskVector + 预测结果     （然后取TOP N  相当于去掉了本来就是1的item  拿到了真实有用的预测item）
'''


def to_vectors(train_set, user_count, item_count, user_list_test, mode):
    # assume that the default is itemBased

    test_mask_dict = defaultdict(lambda: [0] * item_count)
    batch_count = user_count  # 改动  直接写成userCount
    if mode == "itemBased":  # 改动  itemCount userCount互换   batchCount是物品数
        user_count = item_count
        item_count = batch_count
        batch_count = user_count

    train_dict = defaultdict(lambda: [0] * item_count)

    for user_id, i_list in train_set.items():
        for item_id in i_list:
            test_mask_dict[user_id][item_id] = -99999
            if mode == "userBased":
                train_dict[user_id][item_id] = 1.0
            else:
                train_dict[item_id][user_id] = 1.0

    train_vector = []
    for batchId in range(batch_count):
        train_vector.append(train_dict[batchId])

    test_mask_vector = []
    for user_id in user_list_test:
        test_mask_vector.append(test_mask_dict[user_id])

    print("Converting to vectors done....")

    return (torch.Tensor(train_vector)), torch.Tensor(test_mask_vector), batch_count


def get_item_count(file_name):
    item_count = 0
    for line in open(file_name):
        L = line.split(",")
        item_count = max(item_count, int(L[1]))
    return item_count


if __name__ == "__main__":
    train_set, user_count, item_count = load_training_data("data/ml-100k/u1.base", "\t")
    user_count = 943 + 1
    item_count = 1682 + 1
    testSet, GroundTruth = load_test_data("data/ml-100k/u1.test", "\t")
    userList_test = list(testSet.keys())
    trainVector, testMaskVector, batchCount = to_vectors(train_set, user_count, item_count, userList_test, "userBased")
