import numpy as np

from case3 import F1, findNetNodes, F2
from plots import xyz_plot


def Sfun(r):
    return 4 * r[0] ** 2 + r[2] ** 2 - 16


if __name__ == '__main__':
    np.set_printoptions(precision=3)

    P1 = [0, 0, 1, 3]  # P1: z + 3 = 0
    P2 = [0, 4, -1, 0]  # P2: 4*y - z = 0
    P3 = [0, 0, 1, -3]  # P3: z - 3 = 0
    P4 = [0, 1, 0, 4]  # P4: y - 4 = 0

    # OK
    XYZmat = findNetNodes(Sfun, P1, P2, P3, P4)
    print(XYZmat)
    # array([[ 1.323,  1.323,  1.323,  2.   ,  2.   ,  2.   ,  1.323,  1.323,  1.323],
    #        [-0.75 ,  1.631,  4.   ,  0.   ,  2.   ,  4.   ,  0.75 ,  2.369,  4.   ],
    #        [-3.   , -3.   , -3.   ,  0.   ,  0.   ,  0.   ,  3.   ,  3.   ,  3.   ]])

    # OK
    uv_coordinates = F1(st=[5, 2], Smin=0, Smax=20, Tmin=-2, Tmax=6)
    print(uv_coordinates)
    # [-0.5, 0.0]

    xyz_coordinates = F2(uv=uv_coordinates, XYZmat=XYZmat)
    print(xyz_coordinates)
    # [2., 1., 0.]

    xyz_plot(XYZmat, F2)
