import numpy as np
from numpy import linalg as LA
from scipy import interpolate

np.set_printoptions(precision=3)
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Модуль функций для выполнения лаб. работы 3 "ИНТЕРПОЛЯЦИЯ".
""" """
"""
Дан массив координат узлов границы плоской фигуры locST = np.array([[S0,T0], [S1,T1],..., [Sn,Tn]]), 
вписанной в некоторый прямоугольник [0;Smax]x[0;Tmax]
Требуется:
Отобразить заданные точки на S1 - четырехугольная часть поверхности S в трехмерном пространстве 
S - задана уравнением Sfun(x,y,z)=0 и ограниченной 4-я плоскостями P1, P2, P3, P4. (Pi: Ax+By+Cz+D=0)
Для этого построить отображение 2-мерных локальных координат точек границ в 3-х мерные глобальные (S,T) --> (x,y,z)  

Например, 
пусть S1 - есть часть поверхности эллиптического цилиндра S, заданного уравнением:  4*x^2 + z^2 = 16
и ограниченной плоскостями:
P1: z + 3 =0
P2: 4*y - z = 0
P3: z - 3 = 0
P4: y - 4 =0
"""


# --------------------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ --------------------------
# Нахождение длины вектора r
def sqnorm(r):
    return np.sqrt(sum(r * r))


# Уравнение плоскости в виде F(x)=0 ( A*x + B*y + C*z + D = 0 )
def Lin3Dfun(r, ABCD):
    S = np.array(r).dot(np.array(ABCD)[0:len(r)]) + ABCD[len(r)]
    return S


# ----------- вычислим расстояние Хаусдорфа между множествами точек Set0, Set1 ------------
# здесь множество точек Set0,1 в виде np.array[[x0,x1,...], [y0, y1,...], ...].T
# т.е. строка Set0,1 - это координаты одной точки из множества
def Hdist1(Set0, Set1):
    dmax = 0
    for p in Set0:
        Set10 = Set1 - p
        Dvect = LA.norm(Set10, axis=1)
        minD = min(Dvect)
        if (dmax < minD): dmax = minD

    return dmax


def HdrfDist(Set0, Set1):
    return max(Hdist1(Set0, Set1), Hdist1(Set1, Set0))


# Функция вычисления 1-мерного сплайн-полинома x(t), на более частом массиве узлов количеством nnew
def calcSpline3(t, x, nnew, kind='cubic'):
    # создаем интерполяционную функцию x(t)
    xtfun = interpolate.interp1d(t, x, kind=kind)
    # генерируем точки на дуге с малым шагом в коичестве nnew
    tnew = np.linspace(t[0], t[len(t) - 1], num=nnew, endpoint=True)
    xnew = xtfun(tnew)
    return xnew


# ----- Рассчитаем точки заданной узлами линии (буквы) с использованием интерполяции ------
# locST - узлы интеполяции, описывающие линию в 2D, newNpoints - кол-во рассчитываемых точек
def calcline2D(locST, newNpoints=61, kind='quadratic', drawindic=False):
    # создаем интерполяционные функции S(t), T(t) и генерируем точки на дуге с малым шагом dt=0.2
    Npoints = locST.shape[1]
    t = np.linspace(0, Npoints - 1, num=Npoints, endpoint=True)
    Snew = calcSpline3(t, locST[0], nnew=newNpoints, kind=kind)
    Tnew = calcSpline3(t, locST[1], nnew=newNpoints, kind=kind)

    # проводим инициализацию параметров отображения
    Smin = round(min(locST[0])) - 1
    Smax = round(max(locST[0])) + 1
    Tmin = round(min(locST[1])) - 1
    Tmax = round(max(locST[1])) + 1

    # рисуем полученные узлы и точки
    if drawindic:
        plt.plot(locST[0], locST[1], 'o', Snew, Tnew, '-')
        plt.legend(['data', kind], loc='best')
        plt.xlim(Smin, Smax)
        plt.ylim(Tmin, Tmax)
        plt.show()

    return np.vstack((Snew, Tnew))


# ----- Нарисуем заданную часть поверхности S1 с использованием отображения uv2xyz ------
# uv2xyz: [-1;1]x[-1;1] --> S1;  # newNpoints - кол-во узлов интерполяции по каждой оси
def calcsurf3D(uv2xyz, XYZmat, newNpoints=50, drawindic=False):
    # формируем новую частую сетку узлов в квадрате [-1;1]x[-1;1]
    u = np.arange(-1.0, 1.0, 2 / newNpoints)
    v = np.arange(-1.0, 1.0, 2 / newNpoints)
    uu, vv = np.meshgrid(u, v)

    # рассчитываем координаты X,Y, Z новых узловых точек, используя сконструированный полином Лагранжа
    uunew = uu.reshape(uu.shape[0] * uu.shape[1], )
    vvnew = vv.reshape(vv.shape[0] * vv.shape[1], )
    znew = uv2xyz([uunew, vvnew], XYZmat)

    # формируем матрицы координат узловых точек
    xx = znew[0].reshape(uu.shape)
    yy = znew[1].reshape(uu.shape)
    zz = znew[2].reshape(uu.shape)

    # отрисовываем полученный кусок поверхности
    if drawindic:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, cmap=cm.viridis)
        ax.set_xlim3d(round(min(znew[0])) - 1, round(max(znew[0])) + 1)
        ax.set_ylim3d(round(min(znew[1])) - 1, round(max(znew[1])) + 1)
        ax.set_zlim3d(round(min(znew[2])) - 1, round(max(znew[2])) + 1)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        plt.show()

    return xx, yy, zz


# ----- Нарисуем букву, заданную узлами locST на заданной части поверхности S1 ----------
# с использованием отображения st2xyz: [a;b]x[c;d] --> S1;  st2xyz(st) = uv2xyz(st2uv(st))
# newNpoints - кол-во узлов интерполяции по каждой оси
def LetterOnSurfplot(XYZpoints, XXYYZZ):
    newX, newY, newZ = XYZpoints
    xx, yy, zz = XXYYZZ
    xnew = xx.reshape((xx.shape[0] * xx.shape[1],))
    ynew = yy.reshape((yy.shape[0] * yy.shape[1],))
    znew = zz.reshape((zz.shape[0] * zz.shape[1],))

    # рисуем полученные узлы и точки
    fig = plt.figure()
    ax = Axes3D(fig)
    # ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, cmap=cm.viridis)  # Spectral
    ax.plot_wireframe(xx, yy, zz, rstride=10, cstride=10)
    ax.scatter(newX, newY, newZ, c="red", marker='o')
    ax.set_xlim3d(round(min(xnew)) - 1, round(max(xnew)) + 1)
    ax.set_ylim3d(round(min(ynew)) - 1, round(max(ynew)) + 1)
    ax.set_zlim3d(round(min(znew)) - 1, round(max(znew)) + 1)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.legend(['surf', 'data'], loc='best')
    plt.show()


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
