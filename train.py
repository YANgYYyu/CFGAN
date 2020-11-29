import torch
import torch.nn as nn
import model
import random
import evaluation

import numpy as np
import matplotlib.pyplot as plt


def paint(x, y1, y2, y3):
    plt.title("CFGAN")
    plt.xlabel('epoch')
    # plt.ylabel('')
    plt.plot(x, y2, "k-o", color='red', label='recall', markersize='0')
    plt.plot(x, y1, "k-o", color='black', label='precision', markersize='0')  # List,List,List

    plt.plot(x, y3, "k-o", color='green', label='ndcg', markersize='0')
    plt.ylim([0, 0.5])
    plt.legend()  # 图例
    plt.rcParams['lines.linewidth'] = 1
    plt.show()


def main(train_set, user_count, item_count, test_set, ground_truth, train_vector, test_mask_vector, batch_count,
         epoch_count,
         pro_ZR, pro_PM, arfa):
    X = []  # 画图数据的保存
    precision_list = []
    recall_list = []
    ndcg_list = []
    G = model.Generator(item_count)
    D = model.Discriminator(item_count)
    G = G.cuda()
    D = D.cuda()
    criterion1 = nn.BCELoss()  # 二分类的交叉熵
    criterion2 = nn.MSELoss(size_average=False)
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0001)

    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0001)

    G_step = 2
    D_step = 2
    batch_size_g = 32
    batch_size_d = 32
    real_label_g = (torch.ones(batch_size_g)).cuda()
    fake_label_g = (torch.zeros(batch_size_g)).cuda()
    real_label_d = (torch.ones(batch_size_d)).cuda()
    fake_label_d = (torch.zeros(batch_size_d)).cuda()
    ZR = []
    PM = []
    for epoch in range(epoch_count):  # 训练epochCount次

        if epoch % 1 == 0:
            ZR = []
            PM = []
            for i in range(user_count):
                ZR.append([])
                PM.append([])
                ZR[i].append(np.random.choice(item_count, int(pro_ZR * item_count), replace=False))
                PM[i].append(np.random.choice(item_count, int(pro_PM * item_count), replace=False))
        for step in range(D_step):  # 训练D
            # maskVector1是PM方法的体现  这里要进行优化  减少程序消耗的内存

            left_index = random.randint(1, user_count - batch_size_d - 1)
            real_data = (train_vector[left_index:left_index + batch_size_d]).cuda()  # MD个数据成为待训练数据

            mask_vector1 = (train_vector[left_index:left_index + batch_size_d]).cuda()
            for i in range(len(mask_vector1)):
                mask_vector1[i][PM[left_index + i]] = 1
            condition = real_data  # 把用户反馈数据作为他的特征 后期还要加入用户年龄、性别等信息
            real_data_result = D(real_data, condition)
            d_loss_real = criterion1(real_data_result, real_label_d)

            fake_data = G(real_data)
            fake_data = fake_data * mask_vector1
            fake_data_result = D(fake_data, real_data)
            d_loss_fake = criterion1(fake_data_result, fake_label_d)

            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

        for step in range(G_step):  # 训练G0
            # 调整maskVector2\3
            left_index = random.randint(1, user_count - batch_size_g - 1)
            real_data = (train_vector[left_index:left_index + batch_size_g]).cuda()

            mask_vector2 = (train_vector[left_index:left_index + batch_size_g]).cuda()
            mask_vector3 = (train_vector[left_index:left_index + batch_size_g]).cuda()
            for i in range(len(mask_vector2)):
                mask_vector2[i][PM[i + left_index]] = 1
                mask_vector3[i][ZR[i + left_index]] = 1
            fake_data = G(real_data)
            g_loss2 = arfa * criterion2(fake_data, mask_vector3)
            fake_data = fake_data * mask_vector2
            g_fake_data_result = D(fake_data, real_data)
            g_loss1 = criterion1(g_fake_data_result, real_label_g)
            g_loss = g_loss1 + g_loss2
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

        if epoch % 10 == 0:

            people_amount = len(ground_truth)
            recommend_amount = 10
            index = 0
            precisions = 0
            recalls = 0
            ndcgs = 0
            for testUser in test_set.keys():
                data = (train_vector[testUser]).cuda()

                result = G(data) + (test_mask_vector[index]).cuda()
                index += 1
                precision, recall, ndcg = evaluation.compute_top_n_accuracy(test_set[testUser], result,
                                                                            recommend_amount)
                precisions += precision
                recalls += recall
                ndcgs += ndcg

            precisions /= people_amount
            recalls /= people_amount
            ndcgs /= people_amount

            precision_list.append(precisions)
            recall_list.append(recalls)
            ndcg_list.append(ndcgs)

            X.append(epoch)
            print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f},precision:{},recall:{},ndcg:{}'.format(epoch, epoch_count,
                                                                                                   d_loss.item(),
                                                                                                   g_loss.item(),
                                                                                                   precisions, recalls,
                                                                                                   ndcgs))
            paint(X, precision_list, recall_list, ndcg_list)
    return precision_list
