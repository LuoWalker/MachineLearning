import numpy as np
import pandas as pd
from sklearn.datasets import load_iris  # 鸢尾花
from sklearn.model_selection import train_test_split  # 把数据集分为训练集和测试集
from sklearn.metrics import accuracy_score  # 计算分类预测的准确率

# 处理数据
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)  # 构造表格
df["class"] = iris.target
df["class"] = df["class"].map(
    {0: iris.target_names[0], 1: iris.target_names[1], 2: iris.target_names[2]})
print(df.describe())

x = iris.data
y = iris.target.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=11, stratify=y)  # random_state 随机种子,stratify 测试集中与数据中的y比例保持一致

# 核心算法
def l1_distance(a, b):
    return np.sum(np.abs(a-b), axis=1)  # b只能为行向量，axis：将sum保存成1列,行内相加成1列
def l2_distance(a, b):
    return np.sqrt(np.sum((a-b)**2, axis=1))

# 分类器
class KNN(object):
    # 构造器
    def __init__(self, k_neighbors=1, dist_func=l1_distance):
        self.k_neighbors = k_neighbors
        self.dist_func = dist_func

    #训练模型方法
    def fit(self,x,y):
        self.x_train = x
        self.x_train = y

    #预测模型方法
    def predict(self,x):
        predict_y = np.zeros((x.shape[0],1),dtype=self.predict_y.dtype) #默认为0，zeros( (行,列),类型 )
        for i,x_test in enumerate(x): #enumerate:取元组 序号和值
            pass

        return predict_y
