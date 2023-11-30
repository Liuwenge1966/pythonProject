import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.linalg as linalg


# 这是一个用matplotlib显示风机模型，并且计算无人机进行叶片巡检的动态航线的类
class ShowFengji():

    def __init__(self):
        # 设置风机的基本结构参赛
        self.H_tower = 120.0  # 塔筒高度
        self.L_blade = 80.0  # 叶片长度
        self.L_cabin_front = 8.0  # 机舱前部突出塔筒中心至轮毂中心的长度，一般大约5米
        self.L_cabin_tail = 8  # 机舱尾部突出塔筒中心长度，一般大约8～10米

        # 实际旋转的角度
        self.blade_angle = 90  # 叶片1与X轴正向的夹角，代表绕Y轴逆时针转的角度. th1=0时，1号叶片在水平位置
        self.cabin_angle = 90  # 机舱纵向中心线绕Z轴逆时针旋转的角度，也就是决定了3叶片组成的大平面的朝向

        mpl.rcParams['legend.fontsize'] = 10
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')
        self.ax.set_xlim([-100, 100])
        self.ax.set_ylim([-100, 100])
        self.ax.set_zlim([-120, 90])

        self.ax.plot()
        plt.show()

        # 计算风机的各个子模型的实际位置并画出
        # self.ShowAll()

    # 旋转矩阵 欧拉角
    def rotate_mat(self, axis, radian):
        return linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))

    # 绕轴旋转，对于风机来说，不存在绕X轴的旋转
    def rotate(self, p, yaw_Y=0, yaw_Z=0):
        """
        绕轴旋转
        :param p: 原始点三维坐标
        :param yaw_Y: 绕 X轴旋转的角度
        :param yaw_Z: 绕 X轴旋转的角度
        :return: 旋转后的点三维坐标
        """
        # 分别是x,y和z轴,也可以自定义旋转轴
        axis_x, axis_y, axis_z = [1, 0, 0], [0, 1, 0], [0, 0, 1]

        # 旋转角度,转成弧度
        radian_y = -yaw_Y * math.pi / 180
        radian_z = yaw_Z * math.pi / 180
        # 返回绕Y轴旋转矩阵, 相当于叶片旋转
        rot_matrix = self.rotate_mat(axis_y, radian_y)
        newP = np.dot(rot_matrix, p)
        # 返回绕Z轴旋转矩阵，相当于风机旋转
        rot_matrix = self.rotate_mat(axis_z, radian_z)
        newP = np.dot(rot_matrix, newP)
        return newP

    def ShowAll(self):

        # 定义风机的标准模型
        # 塔筒
        self.m_tower = np.array([[0, 0, 0], [0, 0, - self.H_tower]])