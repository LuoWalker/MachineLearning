# %%引入依赖
from scipy.spatial.distance import cdist  # 引入scipy中的距离函数，默认欧氏距离
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, k_means
from sklearn.datasets._samples_generator import make_blobs

# %%导入数据
# x为坐标点，y为类别
x, y = make_blobs(n_samples=100, centers=6, random_state=1234, cluster_std=0.6)
plt.scatter(x[:, 0], x[:, 1], c=y)


# %%算法实现
class K_Means(object):
    # 构造器初始参数：n_cluster(K)、max_iter迭代次数、centroids初始质心
    def __init__(self, n_cluster=6, max_iter=300, centroids=[]):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.centroids = np.array(centroids, dtype=np.float32)  # 数据类型转换

    # 训练模型方法，聚类过程
    def fit(self, data):
        # 没有质心，随机选取质心
        if(self.centroids.shape == (0,)):
            # 随机选择，randint(起,止,个数)，随机生成范围内的整数
            self.centroids = data[np.random.randint(
                0, data.shape[0], self.n_cluster), :]
        for i in range(self.max_iter):
            # 计算到各个质心的距离
            distances = cdist(data, self.centroids)  # 100×6的矩阵
            # 分到最近的质心的类
            c_index = np.argmin(distances, axis=1)
            # 求均值，更新质心
            for i in range(self.n_cluster):
                if i in c_index:
                    self.centroids[i] = np.mean(data[c_index == i], axis=0)

    # 预测方法模型
    def predict(self, samples):
        distances = cdist(samples, self.centroids)  # 100×6的矩阵
        c_index = np.argmin(distances, axis=1)
        return c_index

# %%测试
def plotKMeans(x,y,centroids,subplot,title):
    plt.subplot(subplot)
    plt.scatter(x[:,0],x[:,1],c="r")
    plt.scatter(centroids[:,0],centroids[:,1],c=np.array(range(6)),s=100) #s=size
    plt.title(title)

k_means=K_Means(n_cluster=6, max_iter=300, centroids=np.array([[2,1],[2,2],[2,3],[2,4],[2,5],[2,6]]))
plt.figure(figsize=(16,6))
plotKMeans(x,y,k_means.centroids,121,"初始状态")#121：一行两列的子图中的第一个
plt.show()
k_means.fit(x)
plotKMeans(x,y,k_means.centroids,122,"最终状态")#121：一行两列的子图中的第二个
plt.show()
# %%预测
