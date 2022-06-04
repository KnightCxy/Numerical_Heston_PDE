"""
Program: solving Heston PDE numerically
Author: cai
Date: 2022-04-27
"""
import math
import numpy as np
from scipy.misc import derivative
from scipy import interpolate
import matplotlib.pyplot as plt


class HestonPDE:

    def __init__(self, kappa, theta, rho, sigma, r, q, K, T, smin, smax, vmin, vmax, N, M, L):
        # heston parameter
        self.kappa = kappa
        self.theta = theta
        self.rho = rho
        self.sigma = sigma
        self.r = r
        self.q = q
        self.K = K
        self.T = T
        # transformation parameter
        self.smin = smin
        self.smax = smax
        self.vmax = vmax
        self.vmin = vmin
        self.alpha = self.smax - self.smin  # uniform grid
        self.beta = self.vmax - self.vmin  # uniform grid
        # grid parameter
        self.N = N
        self.M = M
        self.L = L

    # coordinate transformation
    def S_new(self, ksi, order):
        c1 = (math.asinh((self.smax - self.K) / self.alpha))
        c2 = (math.asinh((self.smin - self.K) / self.alpha))
        re = self.K + self.alpha * np.sinh(c1 * ksi + c2 * (1 - ksi))
        re2 = lambda ksi: self.K + self.alpha * np.sinh(c1 * ksi + c2 * (1 - ksi))
        if order == 0:
            return re
        elif order == 1:
            return derivative(re2, ksi, dx=1e-10, n=1)
        elif order == 2:
            return derivative(re2, ksi, dx=1e-10, n=2)
        else:
            return False

    def v_new(self, eta, order):
        d = math.asinh(self.vmax / self.beta)
        re = self.beta * np.sinh(d * eta)
        re2 = lambda eta: self.beta * np.sinh(d * eta)
        if order == 0:
            return re
        elif order == 1:
            return derivative(re2, eta, dx=1e-10, n=1)
        elif order == 2:
            return derivative(re2, eta, dx=1e-10, n=2)
        else:
            return False

    # create grids for s, v, and T in the new coordinate
    def get_grids(self):
        ksi_lst = np.linspace(0, 1, self.N + 1)
        eta_lst = np.linspace(0, 1, self.M + 1)
        tau_lst = np.linspace(0, T, self.L + 1)
        return ksi_lst, eta_lst, tau_lst

    def get_grids2(self, grid_type):
        K = self.K
        Smax = self.smax
        Smin = self.smin
        Vmax = self.vmax
        Vmin = self.vmin
        NS = self.N
        NV = self.M
        if grid_type == 'uniform':
            ds = (Smax - Smin) / (NS - 1)
            dv = (Vmax - Vmin) / (NV - 1)
            Spot = np.arange(Smin, Smax + ds, ds)
            Vol = np.arange(Vmin, Vmax + dv, dv)
            Spot = Spot[:NS]
            Vol = Vol[:NV]
        else:
            C = K / 5
            dz = 1 / (NS - 1) * (math.asinh((Smax - K) / C) - math.asinh(-K / C))
            # The Spot Price Grid
            Z = np.zeros(NS)
            Spot = np.zeros(NS)
            for i in range(NS):
                Z[i] = math.asinh(-K / C) + (i - 1) * dz
                Spot[i] = K + C * math.sinh(Z[i])
            # The volatility grid
            d = Vmax / 10
            dn = math.asinh(Vmax / d) / (NV - 1)
            N = np.zeros(NV)
            Vol = np.zeros(NV)
            for j in range(NV):
                N[j] = (j - 1) * dn
                Vol[j] = d * math.sinh(N[j])
        return (Spot, Vol)

    # construct A matrix
    def A_element(self, i, j):
        """
        i = 1, ..., N + 1
        j = 1,...., M + 1
        :return:
        """
        ksi_lst, eta_lst, tau_lst = self.get_grids()
        S = self.S_new(ksi_lst, 0)
        dSdksi1 = self.S_new(ksi_lst, 1)
        dSdksi2 = self.S_new(ksi_lst, 2)
        v = self.v_new(eta_lst, 0)
        dvdeta1 = self.v_new(eta_lst, 1)
        dvdeta2 = self.v_new(eta_lst, 2)
        dtau = self.T / self.L
        dksi = 1 / self.N
        deta = 1 / self.M
        a = ((dtau*self.rho*self.sigma)/(4*dksi*deta))*v[j-1]*S[i-1]*(1/(dSdksi1[i-1]*dvdeta1[j-1]))
        b = (self.sigma**2*dtau/(2*deta**2))*v[j-1]*(1/(dvdeta1[i-1])**2)-dtau/(2*deta)*(self.kappa*(self.theta-v[j-1])/dvdeta1[j-1]-
              0.5*self.sigma**2*v[j-1]*dvdeta2[j-1]/dvdeta1[j-1]**3)
        c = (dtau/dksi**2)*v[j-1]*(S[i-1]**2)*(1/dSdksi1**2)-dtau/(2*dksi)*((self.r-self.q)*S[i-1]*(1/dSdksi1[i-1])-0.5*v[j-1]*
               S[i-1]**2*(dSdksi2[i-1]/dSdksi1[i-1]**3))
        d = 1+self.r*dtau+(dtau/dksi**2)*v[j-1]*S[i-1]**2*(1/dSdksi1[i-1]**2)+(self.sigma**2*dtau/deta**2)*v[j-1]/dvdeta1[i-1]**2
        e = (dtau/dksi**2)*v[j-1]*S[i-1]**2*(1/dSdksi1[i-1]**2)+dtau/(2*dksi)*((self.r-self.q)*S[i-1]*(1/dSdksi1[i-1])-0.5*v[j-1]*
               S[i-1]**2*(dSdksi2[i-1]/dSdksi1[i-1]**3))
        f = (self.sigma**2*dtau/(2*deta**2))*v[j-1]*(1/(dvdeta1[i-1])**2)-dtau/(2*deta)*(self.kappa*(self.theta-v[j-1])/dvdeta1[j-1]-
              0.5*self.sigma**2*v[j-1]*dvdeta2[j-1]/dvdeta1[j-1]**3)
        return a, b, c, d, e, f


    def HestonimplicitPDE(self, Si, vi):

        NS = self.N
        NV = self.M
        NT = self.L
        Strike = self.K
        r = self.r
        q = self.q
        Smax = self.smax
        Tmin = 0
        Tmax = self.T
        kappa = self.kappa
        theta = self.theta
        sigma = self.sigma
        rho = self.rho

        (Spot, Vol) = self.get_grids2('non-uniform')
        dt = (Tmax - Tmin) / (NT - 1)
        U = np.zeros((NS, NV))
        # Temporary grid for previous time steps
        u = np.zeros((NS, NV))
        # Boundary condition for Call Option at t = Maturity
        for j in range(NV):
            U[:, j] = np.maximum(Spot - Strike, 0)
        for tt in range(NT):
            # Boundary condition for Smin and Smax
            U[0, :] = 0
            U[NS - 1, :] = np.max(Smax - Strike, 0)
            # Boundary condition for Vmax
            U[:, NV - 1] = np.maximum(Spot - Strike, 0)
            # Update the temporary grid u(s,t) with the boundary conditions
            u = U
            # Boundary condition for Vmin.
            # Previous time step values are in the temporary grid u(s,t)
            for ss in range(1, NS - 1):
                DerV = (u[ss, 1] - u[ss, 0]) / (Vol[1] - Vol[0])
                DerS = (u[ss + 1, 0] - u[ss - 1, 0]) / (Spot[ss + 1] - Spot[ss - 1])
                LHS = -r * u[ss, 0] + (r - q) * Spot[ss] * DerS + kappa * theta * DerV
                U[ss, 0] = LHS * dt + u[ss, 0]
            u = U
            for s in range(1, NS - 1):
                for v in range(1, NV - 1):
                    DerS = (u[s + 1, v] - u[s - 1, v]) / (Spot[s + 1] - Spot[s - 1])
                    DerV = (u[s, v + 1] - u[s, v - 1]) / (Vol[v + 1] - Vol[v - 1])
                    DerSS = ((u[s + 1, v] - u[s, v]) / \
                             (Spot[s + 1] - Spot[s]) - (u[s, v] - u[s - 1, v]) / \
                             (Spot[s] - Spot[s - 1])) / (Spot[s + 1] - Spot[s])  # d2U/dS2
                    DerVV = ((u[s, v + 1] - u[s, v]) / \
                             (Vol[v + 1] - Vol[v]) - (u[s, v] - u[s, v - 1]) / \
                             (Vol[v] - Vol[v - 1])) / (Vol[v + 1] - Vol[v])  # d2U/dV2
                    DerSV = (u[s + 1, v + 1] - u[s - 1, v + 1] - u[s + 1, v - 1] + u[s - 1, v - 1]) / \
                            (Spot[s + 1] - Spot[s - 1]) / (Vol[v + 1] - Vol[v - 1])  # d2U/dSdV
                    L = 0.5 * Vol[v] * Spot[s] * Spot[s] * DerSS + rho * sigma * Vol[v] * Spot[s] * DerSV \
                        + 0.5 * sigma * sigma * Vol[v] * DerVV - r * u[s, v] \
                        + (r - q) * Spot[s] * DerS + kappa * (theta - Vol[v]) * DerV
                    U[s, v] = L * dt + u[s, v]
        U = U.transpose()
        f = interpolate.interp2d(Spot, Vol, U)
        return U, f(Si, vi)[0]


if __name__ == "__main__":

    N = 50
    M = 25
    L = 1000
    T = 0.5
    kappa = 2
    theta = 0.07844994
    sigma = 0.74456024
    v0 = 0.0283379
    rho = -0.82445271

    K = 4552.48
    smin = 0
    smax = 2*K
    vmin = 0.001
    vmax = 0.9
    r = 0.01
    q = 0

    option = HestonPDE(kappa, theta, rho, sigma, r, q, K, T, smin, smax, vmin, vmax, N, M, L)
    s_lst = np.linspace(smin, smax, 50)
    vol_lst = np.linspace(vmin, vmax, 25)
    xx, yy = np.meshgrid(s_lst, vol_lst)

    zz,f = option.HestonimplicitPDE(4000, 0.3)
    # for i in range(len(s_lst)):
    #     for j in range(len(vol_lst)):
    #         zz[i][j] = option.HistoricalPDE(s_lst[i], vol_lst[j])
    # print(zz)
    # print(len(option.HestonimplicitPDE(4000, 0.3)))
    fig = plt.figure()
    ax3 = plt.axes(projection='3d')
    # ax3.plot_surface(xx, yy, zz, cmap='rainbow')
    ax3.plot_surface(xx, yy, zz, alpha=0.6, rstride=1, cstride=1, cmap='rainbow')
    ax3.set_xlabel('spot price')
    ax3.set_ylabel('volatility')
    ax3.set_zlabel('Euro call price')
    plt.show()





