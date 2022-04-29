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
    x, y, test_size=0.3, random_state=35, stratify=y)  # random_state 随机种子,stratify 测试集中与数据中的y比例保持一致

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

    # 训练模型方法
    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

    # 预测模型方法
    def predict(self, x):
        # 默认为0，zeros( (行,列),类型 )
        predict_y = np.zeros((x.shape[0], 1), dtype=self.y_train.dtype)
        for i, x_test in enumerate(x):  # enumerate:取元组 序号和值

            # 计算每个测试数据到各个训练数据的距离
            distances = self.dist_func(self.x_train, x_test)
            # 将距离排序,取得其索引
            nn_index = np.argsort(distances)
            # 取前k个，分析其分类
            nn_y = self.y_train[nn_index[:self.k_neighbors]
                                ].ravel()  # ravel()扩展成一维
            # 统计出现最多的分类
            # bincount：统计数组中每个值出现的次数// argmax：返回最大值的索引
            predict_y[i] = np.argmax(np.bincount(nn_y))

        return predict_y

knn = KNN(k_neighbors=3)
knn.fit(x_train, y_train)
result_list = []
for p in [1,2]:
    knn.dist_func = l1_distance if p==1 else l2_distance
    for k in range(1,10,2):
        knn.k_neighbors = k
        predict_y = knn.predict(x_test)
        accurancy = accuracy_score(y_test, predict_y)
        result_list.append([k,"欧式距离" if p==1 else "曼哈顿距离",accurancy])
df = pd.DataFrame(result_list,columns=['k','距离函数','准确率'])
print(df)     