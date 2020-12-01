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

"""
- u.data文件包含了100,000条评分信息，每条记录的形式：user id | item id | rating | timestamp.（分隔符是一个tab）
- u1.base和u1.test是一组训练集和测试集，u1到u5是把u.data分成了5份（用于五折交叉验证实验）。可以通过运行mku.sh重新生成一组u1到u5(原来的会被覆盖)
- ua和ub是把u.data分成了两份。每一份又分成了训练集和测试集。同样可以通过mku.sh重新生成一组ua和ub
- mku.sh文件， 每运行一次，就会随机生成一组u1--u5、ua、ub的数据集。（所以非必要不要用，不然每次实验的数据都不一样）
"""


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


def load_training_data(train_file, split_mark):
    """
    加载训练集
    :param train_file: str 训练集的完整路径
    :param split_mark: str 文件中每一行中的分隔符
    :return: train_set: {'user_id': [item_id, item_id]} 字典，key为用户id，value为用户评分过的商品id列表(已排序)
             user_count: int 用户数量，但这不一定为真实的用户数量
             item_count: int 商品数量，但这不一定为真实的商品数量
    """
    # 打印训练文件的相对路径
    print(train_file)

    # 构建一个value为list的字典，即train_set['key']=list
    train_set = defaultdict(list)
    max_u_id = -1
    max_i_id = -1

    for line in open(train_file):
        # 根据分割符将文件的内容进行分割，从左到右，分别是：用户id，商品id，评分，时间戳
        user_id, item_id, rating, _ = line.strip().split(split_mark)

        user_id = int(user_id)
        item_id = int(item_id)

        # 请注意，我们将所有观察到的评分视为隐反馈
        # key为用户id，value为用户评分过的电影组成的列表
        train_set[user_id].append(item_id)

        max_u_id = max(user_id, max_u_id)
        max_i_id = max(item_id, max_i_id)

    # 对每个用户评分过的电影列表进行排序
    for u, i_list in train_set.items():
        i_list.sort()

    # 请注意，这里的user_count和item_count并不能代表真实值，因为这两个值都是取了用户和商品的最大id+1，而用户和商品的id不一定是连续的
    user_count = max_u_id + 1
    item_count = max_i_id + 1

    print(user_count)
    print(item_count)

    print("Training data loading done: %d users, %d items" % (user_count, item_count))

    return train_set, user_count, item_count


def load_test_data(test_file, split_mark):
    """
    加载测试集数据
    :param test_file: 测试文件的相对路径
    :param split_mark: 文件中每一行的分隔符
    :return: test_set: key为用户id，value为用户评分过的电影组成的列表
             ground_truth: 二维矩阵,第一维装着诸多列表,第二维是个列表,值为一个用户所评价过的所有电影的id
    """
    # 构建一个value为list的字典，即train_set['key']=list
    test_set = defaultdict(list)
    for line in open(test_file):
        # 根据分割符将文件的内容进行分割，从左到右，分别是：用户id，商品id，评分，时间戳
        user_id, item_id, rating, _ = line.strip().split(split_mark)
        user_id = int(user_id)
        item_id = int(item_id)

        # 请注意，我们将所有观察到的评分视为隐反馈
        # key为用户id，value为用户评分过的电影组成的列表
        test_set[user_id].append(item_id)

    # 二维矩阵,第一维装着诸多列表,第二维是个列表,值为一个用户所评价过的所有电影的id
    ground_truth = []
    for u, i_list in test_set.items():
        tmp = []
        for j in i_list:
            tmp.append(j)
        ground_truth.append(tmp)

    print("Test data loading done")

    return test_set, ground_truth


def to_vectors(train_set, user_count, item_count, user_list_test, mode):
    """

    :param train_set: 字典，key为用户id，value为用户评分过的商品id列表(已排序)
    :param user_count: 不一定准确
    :param item_count: 不一定准确
    :param user_list_test: 测试数据集中所有的用户id列表
    :param mode: 当前是根据用户推荐商品,还是根据商品推荐用户,只有两个值:userBased和itemBased
    :return: 返回量trainVector test_mask_vector batchCount
    testMaskVector 与trainVector相对应  -99999对应1
    testMaskVector + 预测结果     （然后取TOP N  相当于去掉了本来就是1的item  拿到了真实有用的预测item）
    """
    # assume that the default is itemBased

    # 构造一个默认字典,如果该字典的key不存在时,返回一个有item_count数量的全零列表([0] * item_count:这里构造了一个item_count数量的列表)
    test_mask_dict = defaultdict(lambda: [0] * item_count)

    batch_count = user_count  # 改动  直接写成user_count
    # 如果是根据商品推荐用的模式,则要将对应的值进行更换
    if mode == "itemBased":  # 改动  item_count user_count互换   batch_count是物品数
        user_count = item_count
        item_count = batch_count
        batch_count = user_count

    train_dict = defaultdict(lambda: [0] * item_count)
    print('train_dice:%s' % type(train_dict))
    # 对test_mask_dict和train_dict进行初始化:test_mask_dict的值:如果该用户对商品进行了评分,则为负数,否则还是0
    #                                      train_dict的值:如果用户对该商品进行了评分,则为正值,否则还是0
    #                                      请注意：这两个字典都是二维的，行为用户id列表，列为商品id列表，用户对商品
    for user_id, i_list in train_set.items():
        for item_id in i_list:
            test_mask_dict[user_id][item_id] = -99999
            if mode == "userBased":
                train_dict[user_id][item_id] = 1.0
            else:
                train_dict[item_id][user_id] = 1.0

    # train_vector是二维列表，行数为用户数，列数为商品数，每一列都是用户是否对其评过分，如果评过分，则对应的值为1.0，否则为0
    train_vector = []
    for batchId in range(batch_count):  # 这里有个问题,如果batch_id不是连续的,这不就有错误了吗?
        train_vector.append(train_dict[batchId])

    # test_mask_vector是二维列表，行数为用户数，列数为商品数，每一列都是用户是否对其评过分，如果评过分，则对应的值为-99999，否则为0
    test_mask_vector = []
    for user_id in user_list_test:
        test_mask_vector.append(test_mask_dict[user_id])

    print("Converting to vectors done....")

    return torch.tensor(train_vector), torch.tensor(test_mask_vector), batch_count


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
