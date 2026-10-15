"""用 NumPy 实现 Adam 的参数更新公式，并用人为给定的三步梯度，演示参数如何变化"""

# 没有训练神经网络，也没有定义损失函数，而是把“梯度已经算出来之后，优化器怎么更新参数”单独拿出来展示

import matplotlib.pyplot as plt
import numpy as np

from adam import adam 
from reporting import plot_result


def main():
    initial_parameter = [1.0, -1.0]  # 初始参数信息
    gradients = [[0.10, -0.20], [0.20, -0.10], [-0.05, -0.30]]  # 给定前三步的梯度
    history, updates = adam(initial_parameter, gradients)
    plot_result(history, updates)
    plt.show()


if __name__ == "__main__":
    main()
