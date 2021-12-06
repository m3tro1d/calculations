import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import MinMaxScaler


def append_column(matrix, column):
    return np.append(matrix, column, axis=1)


def extend_data_lin(data):
    new_data = data[:, :-1]
    nn = new_data.shape[1]
    for i in range(nn - 1):
        for j in range(i, nn):
            new_data = append_column(new_data, data[:, [i]] * data[:, [j]])
    new_data = append_column(new_data, new_data[:, [nn - 1]] * new_data[:, [nn - 1]])
    new_data = append_column(new_data, data[:, [-1]])
    return new_data


def scale_x(x_m):
    x_scaler = MinMaxScaler()
    x_scaler.fit(x_m)
    x_m = x_scaler.transform(x_m)
    return x_m


def apply_regression(x_m, y):
    alpha = 0.001
    method = ElasticNet(alpha=alpha, l1_ratio=0.7)
    model = method.fit(x_m, y)
    return model


def get_x_and_y_matrices(new_data):
    x_m = new_data[:, :-1]
    y = new_data[:, -1:]
    return x_m, y


def lin_reg(data):
    new_data = extend_data_lin(data)
    x_m, y = get_x_and_y_matrices(new_data)

    x_m = scale_x(x_m)
    model = apply_regression(x_m, y)

    return model, x_m
