import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.linalg as linalg

mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlim([-100, 100])
ax.set_ylim([-100, 100])
ax.set_zlim([-120, 90])


# 旋转矩阵 欧拉角
def rotate_mat(axis, radian):
    rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
    return rot_matrix


def rotateTransformation(lot, radian):
    """
    三维坐标旋转函数：
    :param lot:为带旋转点坐标
    :param radian:为X、Y、Z轴旋转角度的对应弧度制的数组，[math.radians(绕X轴角度),math.radians(绕y轴角度),math.radians(绕z轴角度)]
    :return:
    """
    rot_matrix_x = linalg.expm(np.cross(np.eye(3), [1, 0, 0] / linalg.norm([1, 0, 0]) * radian[0]))
    l_x = np.dot(rot_matrix_x, lot)
    rot_matrix_y = linalg.expm(np.cross(np.eye(3), [0, 1, 0] / linalg.norm([0, 1, 0]) * radian[1]))
    l_y = np.dot(rot_matrix_y, l_x)
    rot_matrix_z = linalg.expm(np.cross(np.eye(3), [0, 0, 1] / linalg.norm([0, 0, 1]) * radian[2]))
    l_z = np.dot(rot_matrix_z, l_y)
    return l_z


# 绕轴旋转，对于风机来说，不存在绕X轴的旋转
def rotate(p, yaw_Y=0, yaw_Z=0):
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
    rot_matrix = rotate_mat(axis_y, radian_y)
    newP = np.dot(rot_matrix, p)
    # 返回绕Z轴旋转矩阵，相当于风机旋转
    rot_matrix = rotate_mat(axis_z, radian_z)
    newP = np.dot(rot_matrix, newP)
    return newP


# 风机的标准位置的定义：
# 坐标系定义： 右手坐标，X轴指向东，Y轴指向北，Z轴指向天
# 坐标系原点点(0,0,0)，默认在塔筒与机舱纵向轴中线的交叉点
#
# 叶片旋转：绕Y轴旋转，逆时针
# 叶片默认位置： 1号叶片在水平位置，与X轴夹角=0，叶尖指向正东(X轴正方向)，另两个叶片与1号叶片夹角=+-120
#
# 机机舱旋转：绕Y轴旋转，逆时针方向，为正角度。
# 机舱默认位置：机头朝向正北(Y轴的正方向)(此时机舱的旋转角=90)，机尾朝向正南

# 几个常量
H_tower = 120  # 塔筒高度
L_blade = 80  # 叶片长度
L_cabin_front = 8  # 机舱前部突出塔筒中心至轮毂中心的长度，一般大约8米
L_cabin_tail = 10  # 机舱尾部突出塔筒中心长度，一般大约8～10米

# 实际旋转的角度
blade_angle = 90  # 叶片1与X轴正向的夹角，代表绕Y轴逆时针转的角度. th1=0时，1号叶片在水平位置
cabin_angle = 90  # 机舱纵向中心线绕Z轴逆时针旋转的角度，也就是决定了3叶片组成的大平面的朝向

# 定义塔筒、3个叶片，在他们相对于默认位置进行绕Y轴、Z轴旋转后的的三维点阵
# 塔筒
m = np.array([[0, 0, 0], [0, 0, - H_tower]])
x = [x[0] for x in m]
y = [x[1] for x in m]
z = [x[2] for x in m]
ax.plot(x, y, z, label='parametric curve')

# 机舱
# 机舱只能绕Z轴转
# 机舱的前端，它默认位置是在与X轴夹角90的位置
# theta2 = cabin_angle * np.pi / 180  # 绕Z轴角度转弧度
p1 = [0, 0, 0]
# p2_s = [L_cabin_front * np.cos(theta2), L_cabin_front * np.sin(theta2), 0]
p2_s = [L_cabin_front, 0, 0]
p2 = rotate(p2_s, 0, cabin_angle)
m = np.array([p1, p2])
x = [x[0] for x in m]
y = [x[1] for x in m]
z = [x[2] for x in m]
ax.plot(x, y, z, label='parametric curve')

# 这个点，是叶片的根部，也就是轮毂的中心点
CenterPoint = p2

# 机舱尾部标线
# theta2 = (cabin_angle + 180) * np.pi / 180  # 绕Z轴角度转弧度
p1 = [0, 0, 0]
# p2 = [L_cabin_tail * np.cos(theta2), L_cabin_tail * np.sin(theta2), 0]
p2_s = [L_cabin_tail, 0, 0]
p2 = rotate(p2_s, 0, cabin_angle+180)
m = np.array([p1, p2])
x = [x[0] for x in m]
y = [x[1] for x in m]
z = [x[2] for x in m]
ax.plot(x, y, z, label='parametric curve')

# 叶片的标准坐标
p1 = CenterPoint
P2_s = [L_blade, CenterPoint[1], 0]
# 叶片1
p2 = rotate(P2_s, blade_angle,  (cabin_angle-90))
m = np.array([p1, p2])
x = [x[0] for x in m]
y = [x[1] for x in m]
z = [x[2] for x in m]
ax.plot(x, y, z, label='parametric curve')
# 计算叶片长度
d1 = np.sqrt(np.sum(np.square(p2-p1)))
print("叶片1 旋转后 长度:", d1)

# 叶片2
p2 = rotate(P2_s, blade_angle + 120,  cabin_angle-90)
m = np.array([p1, p2])
x = [x[0] for x in m]
y = [x[1] for x in m]
z = [x[2] for x in m]
ax.plot(x, y, z, label='parametric curve')
# 计算叶片长度
d1 = np.sqrt(np.sum(np.square(p2-p1)))
print("叶片2 旋转后 长度:", d1)

# 叶片3
p2 = rotate(P2_s, blade_angle + 120 + 120,  cabin_angle-90)
m = np.array([p1, p2])
x = [x[0] for x in m]
y = [x[1] for x in m]
z = [x[2] for x in m]
ax.plot(x, y, z, label='parametric curve')
# 计算叶片长度
d1 = np.sqrt(np.sum(np.square(p2-p1)))
print("叶片3 旋转后 长度:", d1)

# 求巡检的飞行路线
# 标准的航点数组
points = [5, 15, 25, 35, 45, 55, 65, 75]
UpPoints = ([10, 30, 20], [25, 30, 20], [40, 30, 20], [55, 30, 20], [70, 30, 20], [80, 30, 20])
LowPoints = ([10, 30, -20], [25, 30, -20], [40, 30, -20], [55, 30, -20], [70, 30, -20], [80, 30, -20])

# 计算 叶片1巡检 前缘侧上方 各航点的实际位置
newPoints = np.arange(len(UpPoints) * 3, dtype=np.float32).reshape(len(UpPoints), 3)  # 维数是(3,3)
for i in range(len(UpPoints)):
    newPoints[i] = rotate(UpPoints[i], blade_angle,  (cabin_angle-90))
x = [x[0] for x in newPoints]
y = [x[1] for x in newPoints]
z = [x[2] for x in newPoints]
ax.scatter(x, y, z, c='r', marker='o')

# 计算 叶片1巡检 前缘侧下方 各航点的实际位置
newPoints = np.arange(len(UpPoints) * 3, dtype=np.float32).reshape(len(UpPoints), 3)  # 维数是(3,3)
for i in range(len(LowPoints)):
    newPoints[i] = rotate(LowPoints[i], blade_angle,  (cabin_angle-90))
x = [x[0] for x in newPoints]
y = [x[1] for x in newPoints]
z = [x[2] for x in newPoints]
ax.scatter(x, y, z, c='b', marker='^')

# 计算 叶片2巡检 前缘侧上方 各航点的实际位置
newPoints = np.arange(len(UpPoints) * 3, dtype=np.float32).reshape(len(UpPoints), 3)  # 维数是(3,3)
for i in range(len(UpPoints)):
    newPoints[i] = rotate(UpPoints[i], blade_angle + 120,  (cabin_angle-90))
x = [x[0] for x in newPoints]
y = [x[1] for x in newPoints]
z = [x[2] for x in newPoints]
ax.scatter(x, y, z, c='r', marker='o')

# 计算 叶片2巡检 前缘侧下方 各航点的实际位置
newPoints = np.arange(len(UpPoints) * 3, dtype=np.float32).reshape(len(UpPoints), 3)  # 维数是(3,3)
for i in range(len(LowPoints)):
    newPoints[i] = rotate(LowPoints[i], blade_angle + 120,  (cabin_angle-90))
x = [x[0] for x in newPoints]
y = [x[1] for x in newPoints]
z = [x[2] for x in newPoints]
ax.scatter(x, y, z, c='b', marker='^')

# 计算 叶片3巡检 前缘侧上方 各航点的实际位置
newPoints = np.arange(len(UpPoints) * 3, dtype=np.float32).reshape(len(UpPoints), 3)  # 维数是(3,3)
for i in range(len(UpPoints)):
    newPoints[i] = rotate(UpPoints[i], blade_angle + 120 + 120,  (cabin_angle-90))
x = [x[0] for x in newPoints]
y = [x[1] for x in newPoints]
z = [x[2] for x in newPoints]
ax.scatter(x, y, z, c='r', marker='o')

# 计算 叶片3巡检 前缘侧下方 各航点的实际位置
newPoints = np.arange(len(UpPoints) * 3, dtype=np.float32).reshape(len(UpPoints), 3)  # 维数是(3,3)
for i in range(len(LowPoints)):
    newPoints[i] = rotate(LowPoints[i], blade_angle + 120 + 120,  (cabin_angle-90))
x = [x[0] for x in newPoints]
y = [x[1] for x in newPoints]
z = [x[2] for x in newPoints]
ax.scatter(x, y, z, c='b', marker='^')

ax.set_xlabel('X(--> East)')
ax.set_ylabel('Y(--> North)')
ax.set_zlabel('Z(--> Up)')

plt.show()
