import data
import train

if __name__ == '__main__':
    trainSet, user_count, item_count = data.load_training_data("data/ml-100k/u2.base", "\t")
    user_count = 943 + 1
    item_count = 1682 + 1
    testSet, GroundTruth = data.load_test_data("data/ml-100k/u2.test", "\t")
    userList_test = list(testSet.keys())
    trainVector, testMaskVector, batchCount = data.to_vectors(trainSet, user_count, item_count, userList_test,
                                                              "userBased")
    train.main(trainSet, user_count, item_count, testSet, GroundTruth, trainVector, testMaskVector, batchCount, 1000,
               0.5, 0.7, 0.03)
