#现在有一堆水果，还有这些水果的真实标签（用ground_true_label表示）和机器学习模型的预测类别（用classifier_pred表示），
#计算这个机器学习模型的错误率和精度。

import numpy as np

# 数据集的真实标签           x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20
ground_true_label = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0,  0,  1,  1,  0,  1,  1,  1,  1,  1,  1])
# 分类模型对数据集的预测标签 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20
classifier_pred =   np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0,  0,  1,  1,  1,  1,  1,  1,  1,  0,  1])

def get_score(gt, pred):
    """
    计算错误率和精度
    :参数 gt: 数据集的真实标签
    :参数 pred: 分类模型对数据集的预测标签
    :返回值: accuracy: 精度
            error_rate: 错误率
    """
    # 计算精度
    accuracy = np.mean(gt == pred)
    # 计算错误率
    error_rate = 1 - accuracy
    
    return accuracy, error_rate

acc, er = get_score(ground_true_label, classifier_pred)

print("机器学习模型的错误率：{:.2f}".format(er))
print("机器学习模型的精度：{:.2f}".format(acc))
