#判断肿瘤是良性的还是恶性的，通过从sklearn中导入威斯康星乳腺肿瘤数据集，训练朴素贝叶斯模型，根据病人信息，
#预测其肿瘤是良性还是恶性，即0或者1

#导入数据集拆分工具
from sklearn.model_selection import train_test_split
#导入威斯康星乳腺肿瘤数据集
from sklearn.datasets import load_breast_cancer
#导入高斯朴素贝叶斯模型
from sklearn.naive_bayes import GaussianNB

cancer = load_breast_cancer()
#将数据集的数值和分类目标赋值给X和y
X,y = cancer.data,cancer.target
#使用数据集拆分工具拆分为训练集和测试集
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=38)

def gaussian_bayes():
    '''
    构建高斯贝叶斯分类模型
    返回值：高斯朴素贝叶斯模型
    '''
    #使用高斯朴素贝叶斯拟合数据
    gnb = GaussianNB()
    #训练模型
    gnb.fit(X_train, y_train)
    return gnb
