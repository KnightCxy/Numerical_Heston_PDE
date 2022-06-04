"""
Program:
Author: cai
Date: 2022-04-27
"""
import math
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize


def Heston_call(kappa, theta, sigma, rho, v0, r, T, s0, K):
    temp1 = p1(kappa, theta, sigma, rho, v0, r, T, s0, K)
    temp2 = p2(kappa, theta, sigma, rho, v0, r, T, s0, K)
    return s0 * temp1 - K * np.exp(-r * T) * temp2


def __p(kappa, theta, sigma, rho, v0, r, T, s0, K, status):
    integrand = lambda phi: (np.exp(-1j * phi * np.log(K)) * __f(phi, kappa, theta, sigma, rho, v0, r, T, s0, status) / (1j * phi)).real
    return 0.5 + (1 / np.pi) * quad(integrand, 0, 100)[0]


def p1(kappa, theta, sigma, rho, v0, r, T, s0, K):
    return __p(kappa, theta, sigma, rho, v0, r, T, s0, K, 1)


def p2(kappa, theta, sigma, rho, v0, r, T, s0, K):
    return __p(kappa, theta, sigma, rho, v0, r, T, s0, K, 2)


def __f(phi, kappa, theta, sigma, rho, v0, r, T, s0, status):
    if status == 1:
        u = 0.5
        b = kappa - rho * sigma
    else:
        u = -0.5
        b = kappa
    a = kappa * theta
    x = np.log(s0)
    d = np.sqrt((rho * sigma * phi * 1j - b) ** 2 - sigma ** 2 * (2 * u * phi * 1j - phi ** 2))
    g = (b - rho * sigma * phi * 1j + d) / (b - rho * sigma * phi * 1j - d)
    C = r * phi * 1j * T + (a / sigma ** 2) * (
                (b - rho * sigma * phi * 1j + d) * T - 2 * np.log((1 - g * np.exp(d * T)) / (1 - g)))
    D = (b - rho * sigma * phi * 1j + d) / sigma ** 2 * ((1 - np.exp(d * T)) / (1 - g * np.exp(d * T)))
    return np.exp(C + D * v0 + 1j * phi * x)


if __name__ == "__main__":

    import pandas as pd
    data = pd.read_csv('/Users/cai/python_program/MF796/796project/Euro_option22.4.1.csv', index_col=0)
    print(data)
    s0 = 4552.48
    K = s0 * np.array(data.index) * 0.01
    r = np.append([0.15, 0.37], np.linspace(0.53, 1.09, 4)) * 0.01
    T = [1/12, 2/12, 3/12, 4/12, 5/12, 6/12]

    def target(kappa, theta, sigma, rho, v0, r, T, s0, K, data):
        re = 0
        for i in range(len(K)):
            for j in range(len(T)):
                re += (Heston_call(kappa, theta, sigma, rho, v0, r[j], T[j], s0, K[i]) - data.iloc[i, j]) ** 2
        return re

    func = lambda x: target(x[0], x[1], x[2], x[3], x[4], r, T, s0, K, data)
    x0 = [0.25, 0.1, 1.5, -0.5, 0.2]
    bound = [(0.0, 2.0), (0.0, 1.0), (0.001, 5.0), (-1, 1), (0.0, 2.0)]
    param = minimize(func, x0, bounds=bound)
    print(param)

    # lst = [1.5, 0.1, 0.25, -0.5, 0.1]
    # bound = [(0.001, 5.0), (0.0, 2.0), (0.0, 2.0), (-1, 1), (0.0, 1.0)]
    # self.sigma = lst[0]
    # self.eta0 = lst[1]
    # self.kappa = lst[2]
    # self.rho = lst[3]
    # self.theta = lst[4]







