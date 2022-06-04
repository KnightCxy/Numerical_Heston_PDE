"""
Program:
Author: cai
Date: 2022-05-01
"""
from numerical_PDE import HestonPDE
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from scipy import interpolate


def BSCallPrice(sigma, S_0, K, r, T, q):
    """ obtain European call option price using the Black-Scholes formula """

    sigmaRtT = (sigma * np.sqrt(T))
    rSigTerm = ((r - q + 0.5 * sigma ** 2) * T)
    d1 = (np.log(S_0 / K) + rSigTerm) / sigmaRtT
    d2 = d1 - sigmaRtT
    term1 = (S_0 * norm.cdf(d1, 0, 1))
    term2 = (K * np.exp(-r * T) * norm.cdf(d2, 0, 1))
    call = term1 - term2
    return call


if __name__ == "__main__":

    N = 50
    M = 25
    L = 1000
    # T = 0.5
    kappa = 2
    theta = 0.07844994
    sigma = 0.74456024
    v0 = 0.0283379
    rho = -0.82445271

    K = 4552.48
    smin = 0
    smax = 2 * K
    vmin = 0.001
    vmax = 0.9
    # r = 0.01
    q = 0
    #
    # option = HestonPDE(kappa, theta, rho, sigma, r, q, K, T, smin, smax, vmin, vmax, N, M, L)
    # s_lst = np.linspace(smin, smax, 50)
    # vol_lst = np.linspace(vmin, vmax, 25)
    # xx, yy = np.meshgrid(s_lst, vol_lst)
    #
    # zz = option.HestonimplicitPDE()

    # examine the difference between market price and model price
    data = pd.read_csv('/Users/cai/python_program/MF796/796project/Euro_option22.4.1.csv', index_col=0)
    # print(data)
    im_vol = np.zeros((11,6))
    # print(len(im_vol))
    K = 4552.48 * np.array(data.index) * 0.01
    T = [1 / 12, 2 / 12, 3 / 12, 4 / 12, 5 / 12, 6 / 12]
    r = np.append([0.15, 0.37], np.linspace(0.53, 1.09, 4)) * 0.01
    data = np.array(data)
    for i in range(len(K)):
        for j in range(len(T)):
            func = lambda x: (BSCallPrice(x, 4552.48, K[i], r[j], T[j], 0) - data[i][j])**2
            opti = minimize(func, 0.5)
            im_vol[i][j] = opti.x[0]
    print(im_vol)

    model_price = np.zeros((11,6))
    abs_diff = np.zeros((11, 6))
    for i in range(len(K)):
        for j in range(len(T)):
            option = HestonPDE(kappa, theta, rho, sigma, r[j], q, K[i], T[j], smin, smax, vmin, vmax, N, M, L)
            temp,f = option.HestonimplicitPDE(4552.48, im_vol[i][j])
            model_price[i][j] = f
            abs_diff[i][j] = abs(f-data[i][j])
    print('the model price is:')
    print(model_price)
    print('the difference is:')
    print(abs_diff)








