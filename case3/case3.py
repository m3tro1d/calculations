import numpy as np
from scipy.optimize import root


def transform1(t, t_min, t_max):
    y = (t - t_min) / (t_max - t_min)
    return 2 * y - 1


def generate_linear_trans(min1, max1, min2, max2):
    def linear_trans(xy):
        u = transform1(xy[0], t_min=min1, t_max=max1)
        v = transform1(xy[1], t_min=min2, t_max=max2)
        return u, v

    return linear_trans


def F1(st, Smin, Smax, Tmin, Tmax):
    linear_trans = generate_linear_trans(Smin, Smax, Tmin, Tmax)
    return linear_trans(st)


def calcPuv3x3(u, v):
    Plist = list()

    Plist.append(u * (u - 1) * v * (v - 1) / 4)  # P(u=-1, v=-1)
    Plist.append(-(u + 1) * (u - 1) * v * (v - 1) / 2)  # P(u=0 , v=-1)
    Plist.append(u * (u + 1) * v * (v - 1) / 4)  # P(u=1 , v=-1)

    Plist.append(-u * (u - 1) * (v + 1) * (v - 1) / 2)  # P(u=-1, v=0)
    Plist.append((u + 1) * (u - 1) * (v - 1) * (v + 1))  # P(u=0 , v=0)
    Plist.append(-u * (u + 1) * (v + 1) * (v - 1) / 2)  # P(u=1 , v=0)

    Plist.append(u * (u - 1) * v * (v + 1) / 4)  # P(u=-1 , v=1)
    Plist.append(-(u + 1) * (u - 1) * v * (v + 1) / 2)  # P(u=0 , v=1)
    Plist.append(u * (u + 1) * v * (v + 1) / 4)  # P(u=1 , v=1)

    return np.array(Plist)


def get_polinom3x3(f_vect):
    def calcP3(uv, z_vect=f_vect):
        p_vect = calcPuv3x3(uv[0], uv[1])
        return z_vect @ p_vect

    return calcP3


def F2(uv, XYZmat):
    x_values = XYZmat[0]
    x_polynomial = get_polinom3x3(x_values)

    y_values = XYZmat[1]
    y_polynomial = get_polinom3x3(y_values)

    z_values = XYZmat[2]
    z_polynomial = get_polinom3x3(z_values)

    return x_polynomial(uv), y_polynomial(uv), z_polynomial(uv)


def findPoint(Sfun, P1, P2):
    def func(r):
        return np.array([
            Sfun(r),
            np.dot(P1[:3], r) + P1[-1],
            np.dot(P2[:3], r) + P2[-1],
        ])

    return root(func, np.array([1.0, 1.0, 1.0])).x


def findCorners4(Sfun, P1, P2, P3, P4):
    return (
        findPoint(Sfun, P4, P1),
        findPoint(Sfun, P3, P4),
        findPoint(Sfun, P2, P3),
        findPoint(Sfun, P1, P2),
    )


def meanPlane(ABCD1, ABCD2, meanpoint):
    plane_norm = np.cross(ABCD1[:3], ABCD2[:3])

    return np.array([
        plane_norm[0],
        plane_norm[1],
        plane_norm[2],
        np.dot(plane_norm, meanpoint)
    ])


def findNetNodes(Sfun, P1, P2, P3, P4):
    r00, r06, r08, r02 = findCorners4(Sfun, P1, P2, P3, P4)

    meanpoint = (r00 + r02 + r06 + r08) / 4

    P5 = meanPlane(ABCD1=P1, ABCD2=P3, meanpoint=meanpoint)
    P6 = meanPlane(ABCD1=P2, ABCD2=P4, meanpoint=meanpoint)

    r03 = findPoint(Sfun, P5, P2)
    r05 = findPoint(Sfun, P5, P4)
    r01 = findPoint(Sfun, P1, P6)
    r07 = findPoint(Sfun, P3, P6)

    r04 = findPoint(Sfun, P5, P6)

    XYZmat = np.vstack((r00, r01, r02, r03, r04, r05, r06, r07, r08)).T
    return XYZmat
