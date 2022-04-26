#多元线性回归
import numpy as np
import matplotlib.pyplot as plt

# 导入数据
points = np.genfromtxt("data.csv", delimiter=",")
x = points[:, 0]
y = points[:, 1]

# 绘制散点图
plt.scatter(x, y)
plt.show()

# 损失函数
def compute_cost(w, b, points):
    total_cost = 0
    M = len(points)                 # 点的个数
    for i in range(M):
        x = points[i, 0]
        y = points[i, 1]
        total_cost += (y-w*x-b)**2  # 损失函数
    return total_cost/M

# 拟合函数
def average(data):  # 求平均值
    sum = 0
    num = len(data)
    for i in range(num):
        sum += data[i]
    return sum/num
def fit(x, y):
    M = len(x)
    x_bar = average(x)
    sum_yx = 0
    sum_x2 = 0
    sum_mb = 0
    for i in range(M):
        sum_yx += y[i]*(x[i]-x_bar)
        sum_x2 += x[i]**2
    w = sum_yx/(sum_x2-M*(x_bar**2))  # 计算w
    for i in range(M):
        sum_mb += (y[i]-w*x[i])
    b = sum_mb/M  # 计算b
    return w, b


# 输出参数
w, b = fit(x, y)
cost = compute_cost(w, b, points)
print("w is:", w)
print("b is:", b)
print("cost is:", cost)

# 画出拟合图像
plt.scatter(x, y)
predict_y = w*x+b
plt.plot(x, predict_y, c="r")
plt.show()
