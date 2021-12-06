# ====================== вспомогательный модуль для отрисовки и анализа траектотрий ====================
''' '''
import numpy as np
import math
import matplotlib.pyplot as plt

def vnorm2(r):
    return np.sqrt(np.sum(r*r))

def cntwisevel(r):
    r1 = np.copy(r)[::-1]
    r1[0] = - r1[0]
    return r1/vnorm2(r1)

# ----------------------------------------------------------------------------------------------------
# -------------   функция отрисовки динамики ошибки управления и управляющих воздействий -------------
# S_trace[0] - временная последов-ть, S_trace[1] - последовательность ошибок системы,
# U_trace[0] - временная последовательность, U_trace[1:3] - последовательность регулировок wL, wR
# Pars - параметры управления и достижения заданной цели - для более информативной отрисовки
# ----------------------------------------------------------------------------------------------------
def drawErrorDynamic(S_trace, U_trace, Pars):
    eps_r = Pars['eps_r']
    if 'Tdone' in Pars:
        Tdone = Pars['Tdone']
    else:
        Tdone = -1

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].plot(S_trace[0], S_trace[1], 'c', label='расстояние до цели')
    axes[0].plot([S_trace[0][0], S_trace[0][-1]], [eps_r, eps_r], 'r--', label='цель')
    axes[0].set_xlabel('t, сек')
    axes[0].set_ylabel('err_r, м')
    minV = min(S_trace[1]); maxV = max(S_trace[1])
    if Tdone > 0:
        axes[0].plot([Tdone, Tdone], [minV, maxV], 'y--', label='время достижения цели={:05.1f}'.format(Tdone))
    axes[0].legend(fontsize=12)

    axes[0].set_title('Динамика ошибки управления')
    axes[1].plot(U_trace[0], U_trace[1], 'g-.', label='Угловая скорость левого мотора')
    axes[1].plot(U_trace[0], U_trace[2], 'b--', label='Угловая скорость правого мотора')
    axes[1].set_xlabel('t, сек')
    axes[1].set_ylabel('wL,wR, рад/с')
    axes[1].legend(fontsize=12)

# ----------------------------------------------------------------------------------------------------
# -------------   функция отрисовки динамики ошибок управления и самого управления      --------------
# U_trace[0] - временная последовательность, U_trace[1:3] - последовательность регулировок wL, wR
# U_trace[5:9] - последовательность ошибок fi_err, r_err, fi_sumerr, r_sumerr
# Pars - параметры границ управления и точности достижения цели - для более информативной отрисовки
# ----------------------------------------------------------------------------------------------------
def drawErrorDynamic2(U_trace, Pars):
    eps_r = Pars['eps_r']
    if 'Tdone' in Pars: Tdone = Pars['Tdone']
    else: Tdone = -1
    if 'Wmin' in Pars: Wmin = Pars['Wmin']
    else: Wmin = 0
    if 'Wmax' in Pars: Wmax = Pars['Wmax']
    else: Wmax = 0

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes[0][0].set_title('Динамика расстояния до цели')
    axes[0][0].plot(U_trace[0], U_trace[6], 'c', label='расстояние до цели')
    axes[0][0].plot([U_trace[0][0], U_trace[0][-1]], [eps_r, eps_r], 'r--', label='цель')
    axes[0][0].set_xlabel('t, сек')
    axes[0][0].set_ylabel('err_r, м')
    minE = 0; maxE = max(U_trace[6])
    if Tdone > 0:
        axes[0].plot([Tdone, Tdone], [minE, maxE], 'y--', label='время достижения цели={:05.1f}'.format(Tdone))
    axes[0][0].legend(fontsize=12)

    axes[0][1].set_title('Динамика ошибки по углу')
    axes[0][1].plot(U_trace[0], U_trace[5], 'g-.', label='Ошибка управления по углу')
    axes[0][1].plot(U_trace[0], U_trace[7], 'b--', label='Суммарная ошибка управления по углу')
    axes[0][1].set_xlabel('t, сек')
    axes[0][1].set_ylabel('fi_error, fi_sumerror, рад')
    axes[0][1].legend(fontsize=12)

    axes[1][1].set_title('Динамика суммарной ошибки по расстоянию')
    axes[1][1].plot(U_trace[0], U_trace[8], 'b--', label='Суммарная ошибка управления по расстоянию до цели')
    axes[1][1].set_xlabel('t, сек')
    axes[1][1].set_ylabel('r_sumerror, м')
    axes[1][1].legend(fontsize=12)

    axes[1][0].set_title('Динамика управляющих воздействий')
    axes[1][0].plot(U_trace[0], U_trace[1], 'g-.', label='Угловая скорость левого мотора')
    axes[1][0].plot(U_trace[0], U_trace[2], 'b--', label='Угловая скорость правого мотора')
    axes[1][0].set_xlabel('t, сек')
    axes[1][0].set_ylabel('wL,wR, рад/с')
    if 'Wmin' in Pars:
        axes[1][0].plot([U_trace[0][0], U_trace[0][-1]], [Wmin, Wmin], 'r--', label='Wmin')
    if 'Wmax' in Pars:
        axes[1][0].plot([U_trace[0][0], U_trace[0][-1]], [Wmax, Wmax], 'r--', label='Wmax')
    axes[1][0].legend(fontsize=12)

# ----------------------------------------------------------------------------------------------------
# -------------     функция отрисовки траектории робота и цели, и динамики vC, wC   ------------------
# S_trace[0] - последовательность координат X, S_trace[1] - последовательность координат Y робота
# G_trace[0] - последовательность координат X, G_trace[1] - последовательность координат Y цели
# U_trace[0] - последовательность шагов времени, U_trace[3:5] - последовательность регулировок u, w
# Pars - параметры управления и достижения заданной цели - для более информативной отрисовки
# ----------------------------------------------------------------------------------------------------
def drawDynamic2D(S_trace, G_trace, U_trace, Pars):
    if 'Tdone' in Pars:
        Tdone = Pars['Tdone']
    else:
        Tdone = -1

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].plot(S_trace[0], S_trace[1], 'c--', marker='*', label='траектория робота')
    axes[0].plot(G_trace[0], G_trace[1], 'r', label='траектория цели')
    axes[0].set_xlabel('x, м')
    axes[0].set_ylabel('y, м')
    axes[0].legend(loc=1, fontsize=12)

    axes[0].set_title('Динамика скорости (м/с)')
    axes[1].plot(U_trace[0], U_trace[3], 'g-.', label='Линейная скорость vC, м/с')
    axes[1].plot(U_trace[0], U_trace[4], 'b--', label='Угловая скорость поворота wC, рад/с')
    axes[1].set_xlabel('t, сек')
    axes[1].set_ylabel('vC, wC')
    minV = min(U_trace[3]); maxV = max(U_trace[3])
    if Tdone > 0:
        axes[1].plot([Tdone, Tdone], [minV, maxV], 'y--', label='время достижения цели={:05.1f}'.format(Tdone))
    axes[1].legend(fontsize=12)

# ----------------------------------------------------------------------------------------------------
# -------------     функция анализа времени достижения заданной точности управления   ----------------
# S_trace[0] - последовательность шагов времени, S_trace[1:3] - последовательность координат X,Y робота
# G_trace[0] - последовательность шагов времени, G_trace[1:3] - последовательность координат X,Y цели
# eps - точность, с которой необходимо достичь цели
# ----------------------------------------------------------------------------------------------------
def analysStrace(S_trace, G_trace, eps):
    Ttrace = S_trace[0]
    dXtrace = (S_trace[1] - G_trace[1])*(S_trace[1] - G_trace[1])
    dYtrace = (S_trace[2] - G_trace[2])*(S_trace[2] - G_trace[2])
    Err_trace = np.sqrt(dXtrace + dYtrace)
    u = Err_trace > eps
    return max(Ttrace[u])

# ----------------------------------------------------------------------------------------------------
# -------------     функция анализа скорости сходимости последовательности скоростей  ----------------
# S_trace[0] - последовательность шагов времени, S_trace[0] - последовательность скоростей,
# Vdir - заданная круизная скорость,   Veps - точность, с которой необходимо достичь цели
# ----------------------------------------------------------------------------------------------------
def analysStrace_2(S_trace, G_trace, eps):
    Ttrace = S_trace[0]
    dXtrace = (S_trace[1] - G_trace[1])*(S_trace[1] - G_trace[1])
    dYtrace = (S_trace[2] - G_trace[2])*(S_trace[2] - G_trace[2])
    Err_trace = np.sqrt(dXtrace + dYtrace)
    u = Err_trace > eps
    y =  Err_trace[u]
    cc = y[1:] / y[:-1]
    return cc

# ----------------------------------------------------------------------------------------------------
# -----------------         функция отрисовки графика оценки скорости сходимости  --------------------
# ----------------------------------------------------------------------------------------------------
def show_convergence_rate_plot(cc):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(cc, 'k--', color='brown', label='последовательность')
    cmean = cc.mean()
    ax.plot([1, len(cc)], [cmean, cmean], 'r--', label='среднее значение')
    plt.title('Оценка скорости сходимости', fontsize=16)
    plt.xlabel('n, шаг', fontsize=14)
    plt.ylabel('c[n]', fontsize=14)
    ax.legend(fontsize=12)
    plt.show()
