# 多元线性回归
from re import M
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


# 初始参数定义
alpha = 0.0001
init_w = 0
init_b = 0
num_iter = 10

# 梯度下降
def gradient_descent(points, init_w, init_b, alpha, num_iter):
    w = init_w
    b = init_b
    list_cost = []  # 记录损失变化

    for i in range(num_iter):
        list_cost.append(compute_cost(w, b, points))
        w, b = step_gradient_descent(w, b, points, alpha)
    return [w, b, list_cost]


def step_gradient_descent(w, b, points, alpha):
    sum_grad_w = 0
    sum_grad_b = 0
    M = len(points)
    for i in range(M):
        x = points[i, 0]
        y = points[i, 1]
        sum_grad_w += (w*x+b-y)*x
        sum_grad_b += w*x+b-y

    grad_w = 2/M*sum_grad_w
    grad_b = 2/M*sum_grad_b
    update_w = w - alpha*grad_w
    update_b = b - alpha*grad_b
    return update_w, update_b

w,b,list_cost=gradient_descent(points,init_w,init_b,alpha,num_iter)
print("w is ",w)
print("b is ",b)
plt.plot(list_cost)
plt.show()

# 画出拟合图像
plt.scatter(x, y)
predict_y = w*x+b
plt.plot(x, predict_y, c="r")
plt.show()