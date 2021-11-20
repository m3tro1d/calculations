# нарисуем все это
import matplotlib.pyplot as plt
from matplotlib import cm

from utils import calcsurf3D


def xyz_plot(XYZmat, F2):
    xx, yy, zz = calcsurf3D(F2, XYZmat)
    # Настраиваем 3D график
    fig = plt.figure(figsize=[9, 6])
    ax = fig.gca(projection='3d')
    # Задаем угол обзора
    ax.view_init(30, -60)
    # Рисуем поверхность
    surf = ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm)
    # нарисуем узлы интерполяции
    ax.scatter(XYZmat[0], XYZmat[1], XYZmat[2], s=100, marker='X')

    ax.set_xlim3d(0, 6)
    ax.set_ylim3d(-3, 3)
    plt.show()
