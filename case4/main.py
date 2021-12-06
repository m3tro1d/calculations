import pandas as pd

from case4 import lin_reg


def read_data():
    data = pd.read_csv('data/online_acad_lin.csv', sep=';')
    return data.values


if __name__ == '__main__':
    data = read_data()
    model, Xnew = lin_reg(data)
