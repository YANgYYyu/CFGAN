import math

'''
衡量预测效果
groundTruth:真实结果
result:预测向量  此量经过了+testMaskVector操作处理  已经剔除掉训练集中用户有过反映的量
topN：取几个预测结果
'''


def compute_top_n_accuracy(ground_truth, result, top_n):
    result = result.tolist()
    for i in range(len(result)):
        result[i] = (result[i], i)
    result.sort(key=lambda x: x[0], reverse=True)
    # print(result)
    hit = 0
    dcg = 0
    idcg = 0
    idcg_count = len(ground_truth)
    for i in range(top_n):
        if result[i][1] in ground_truth:
            hit = hit + 1
            dcg += 1 / math.log2(i + 2)
        if idcg_count > 0:
            idcg += 1 / math.log2(i + 2)
            idcg_count -= 1
    return hit / top_n, hit / len(ground_truth), dcg / idcg
# computeTopNAccuracy(0,[1,2,5,9,100,-5,6,0],0)
