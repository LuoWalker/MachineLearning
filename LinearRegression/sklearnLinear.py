# 一元线性回归
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# 导入数据
points = np.genfromtxt("data.csv", delimiter=",")
x = points[:, 0]
y = points[:, 1]

# 绘制散点图
#plt.scatter(x, y)
#plt.show()

# 损失函数
def compute_cost(w, b, points):
    total_cost = 0
    M = len(points)                 # 点的个数
    for i in range(M):
        x = points[i, 0]
        y = points[i, 1]
        total_cost += (y-w*x-b)**2  # 损失函数
    return total_cost/M


lr = LinearRegression()
x_new = x.reshape(-1, 1)
y_new = y.reshape(-1, 1)

lr.fit(x_new, y_new)
w = lr.coef_[0, 0]  # coefficient 回归系数
b = lr.intercept_[0]  # intercept 截距

cost = compute_cost(w, b, points)
print("w is ", w)
print("b is ", b)
print("cost is ", cost)

plt.scatter(x_new, y_new)
predict_y = w*x+b
plt.plot(x, predict_y, c="r")
plt.show()
