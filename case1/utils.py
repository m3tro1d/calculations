# ====================== вспомогательный модуль для отрисовки и анализа траектотрий ====================
''' '''
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------
# -------------     функция отрисовки динамики скорости и управляющих воздействий --------------------
# V_trace[0] - последовательность шагов времени, V_trace[0] - последовательность скоростей,
# U_trace[0] - последовательность шагов времени, U_trace[1:3] - последовательность регулировок
# Pars - параметры управления и достижения заданной цели - для более информативной отрисовки
# ----------------------------------------------------------------------------------------------------
def drawDynamic(V_trace, U_trace, Pars):
    Vdir = Pars['Vdir']
    if 'Tdone' in Pars:
        Tdone = Pars['Tdone']
    else:
        Tdone = -1

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].plot(V_trace[0], V_trace[1], 'c', label='скорость робота')
    axes[0].plot([V_trace[0][0], V_trace[0][-1]], [Vdir, Vdir], 'r--', label='цель')
    axes[0].set_xlabel('t, сек')
    axes[0].set_ylabel('V, м/с')
    minV = min(V_trace[1]); maxV = max(V_trace[1])
    if Tdone > 0:
        axes[0].plot([Tdone, Tdone], [minV, maxV], 'y--', label='время достижения цели={:05.1f}'.format(Tdone))
    axes[0].legend(loc=4, fontsize=12)

    axes[0].set_title('Динамика скорости (м/с)')
    axes[1].plot(U_trace[0], U_trace[1], 'g-.', label='Пропорциональный регулятор')
    axes[1].plot(U_trace[0], U_trace[2], 'b--', label='Интегральный регулятор')
    axes[1].plot(U_trace[0], U_trace[1] + U_trace[2], 'r.', label='ПИ-регулятор')
    axes[1].set_xlabel('t, сек')
    axes[1].set_ylabel('U, Н')
    axes[1].legend(fontsize=12)

# ----------------------------------------------------------------------------------------------------
# -------------     функция отрисовки динамики скорости и ошибки управления --------------------
# V_trace[0] - последовательность шагов времени, V_trace[0] - последовательность скоростей,
# Pars - параметры управления и достижения заданной цели - для более информативной отрисовки
# ----------------------------------------------------------------------------------------------------
def drawErrorDynamic(V_trace, Pars):
    Vdir = Pars['Vdir']
    if 'Veps' in Pars: Veps = Pars['Veps']
    else:
        Veps = 0.1
    if 'Tdone' in Pars: Tdone = Pars['Tdone']
    else:
        Tdone = -1

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].plot(V_trace[0], V_trace[1], 'c', label='скорость робота')
    axes[0].plot([V_trace[0][0], V_trace[0][-1]], [Vdir, Vdir], 'r--', label='цель')
    axes[0].set_xlabel('t, сек')
    axes[0].set_ylabel('V, м/с')
    minV = min(V_trace[1]); maxV = max(V_trace[1])
    if Tdone > 0:
        axes[0].plot([Tdone, Tdone], [minV, maxV], 'y--', label='время достижения цели={:05.1f}'.format(Tdone))
    axes[0].legend(loc=4, fontsize=12)
    axes[0].set_title('Динамика скорости (м/с)')

    Verror = Vdir - V_trace[1]
    axes[1].plot(V_trace[0], Verror, 'g-.', label='Ошибка управления')
    axes[1].plot([V_trace[0][0], V_trace[0][-1]], [0, 0], 'r', label='цель')
    axes[1].plot([V_trace[0][0], V_trace[0][-1]], [Veps, Veps], 'k--', label='+Veps')
    axes[1].plot([V_trace[0][0], V_trace[0][-1]], [-Veps, -Veps], 'k--', label='-Veps')
    axes[1].set_xlabel('t, сек')
    axes[1].set_ylabel('ошибка управления, м/с')
    axes[1].set_ylim(-20*Veps, 20*Veps)
    minE = min(Verror); maxE = max(Verror)
    if Tdone > 0:
        axes[1].plot([Tdone, Tdone], [minE, maxE], 'y--')
    axes[1].legend(loc=4, fontsize=12)
    axes[1].set_title('Динамика изменения ошибки управления (м/с)')

# ----------------------------------------------------------------------------------------------------
# -------------     функция анализа времени достижения заданной точности управления   ----------------
# V_trace[0] - последовательность шагов времени, V_trace[0] - последовательность скоростей,
# Vdir - заданная круизная скорость,   Veps - точность, с которой необходимо достичь цели
# ----------------------------------------------------------------------------------------------------
def analysVtrace(V_trace, Vdir, Veps):
    Ttrace = V_trace[0]
    u = abs(V_trace[1] - Vdir) > Veps
    return max(Ttrace[u])

# ----------------------------------------------------------------------------------------------------
# -------------     функция анализа скорости сходимости последовательности скоростей  ----------------
# V_trace[0] - последовательность шагов времени, V_trace[0] - последовательность скоростей,
# Vdir - заданная круизная скорость,   Veps - точность, с которой необходимо достичь цели
# ----------------------------------------------------------------------------------------------------
def analysVtrace_2(V_trace, Vdir, Veps):
    Vtrace = V_trace[1]
    u = (Vdir - Vtrace) > Veps
    y =  Vdir - Vtrace[u]
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
