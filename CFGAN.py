import data
import train

if __name__ == '__main__':
    # train_set:字典，key为用户id，value为用户评分过的商品id列表(已排序);user_count:用户数量,不一定准确;item_count:商品数量,不一定准确
    train_set, user_count, item_count = data.load_training_data("data/ml-100k/u2.base", "\t")
    user_count = 943 + 1
    item_count = 1682 + 1
    # test_set: key为用户id，value为用户评分过的电影组成的列表;ground_truth: 二维矩阵,第一维装着诸多列表,第二维是个列表,值为一个用户所评价过的所有电影的id
    test_set, ground_truth = data.load_test_data("data/ml-100k/u2.test", "\t")
    # 测试数据集中所有的用户id列表
    user_list_test = list(test_set.keys())
    train_vector, test_mask_vector, batch_count = data.to_vectors(train_set, user_count, item_count, user_list_test,
                                                              "userBased")
    train.main(train_set, user_count, item_count, test_set, ground_truth, train_vector, test_mask_vector, batch_count, 1000,
               0.5, 0.7, 0.03)
