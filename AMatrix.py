# %%
import numpy as np

def A_Matrix(p_heston, p_option, p_grid):
        N       = p_grid['NS'] # S
        M       = p_grid['NV']  # V
        L       = p_grid['NT']  # T
        T       = p_grid['Tmax'] 
        Smin    = p_grid['Smin']
        Smax    = p_grid['Smax']
        Vmin    = p_grid['Vmin']
        Vmax    = p_grid['Vmax']
        kappa   = p_heston['kappa']
        theta   = p_heston['theta']
        sigma   = p_heston['sigma']
        v0      = p_heston['v0']
        rho     = p_heston['rho']
        lamda   = p_heston['lambda']
        r       = p_option['r']
        q       = p_option['q']

        delta_ksai = 1 / N
        delta_ita = 1 / M
        delta_tau = (T - p_grid['Tmin']) / L

        U_dic, dUdksai, dUdita, dUtau, d2Udksai, d2Udita, d2Utau = {}, {}, {}, {}, {}, {}, {}

        for index in range(L+2):
                U_dic[index], dUdksai[index], dUdita[index], dUtau[index], d2Udksai[index], d2Udita[index], d2Utau[index]\
                = np.zeros((N+1, M+1)), np.zeros((N+1, M+1)), np.zeros((N+1, M+1)), np.zeros((N+1, M+1)), np.zeros((N+1, M+1)), np.zeros((N+1, M+1)), np.zeros((N+1, M+1))
        # for k in range(1, L + 1):
        #         for i in range(N):
        #                 for j in range(M):
        #                         # Central difference for dU/dξ
        #                         dUdksai[k+1][i,j]  = (U_dic[k+1][i+1, j] - U_dic[k+1][i-1, j]) / (2 * delta_ksai)  
        #                         # Central difference for dU/dη 
        #                         dUdita[k+1][i,j]   = (U_dic[k+1][i, j+1] - U_dic[k+1][i, j-1]) / (2 * delta_ita)  
        #                         # Central difference for dU/dτ 
        #                         dUtau[k+1][i,j]    = (U_dic[k+1][i, j] - U_dic[k][i, j]) / (delta_tau)
        #                         # Central difference for d2U/dξ2
        #                         d2Udksai[k+1][i,j] = (U_dic[k+1][i-1, j] - 2*U_dic[k+1][i, j] + U_dic[k+1][i+1, j]) / (delta_ksai**2)  
        #                         # Central difference for d2U/dη2 
        #                         d2Udita[k+1][i,j]  = (U_dic[k+1][i, j-1] - 2*U_dic[k+1][i, j] + U_dic[k+1][i, j+1]) / (delta_ita**2)  
        #                         # Central difference for d2U/dξdη
        #                         d2Utau[k+1][i,j]   = (U_dic[k+1][i-1, j-1] - U_dic[k+1][i-1, j+1] - U_dic[k+1][i+1, j-1]\
        #                                 + U_dic[k+1][i+1, j+1]) / (4 * delta_ita * delta_ksai)  
        #Calculate derivatives for later use
        v = np.linspace(Vmin, Vmax, M)
        S = np.linspace(Smin, Smax, N)
        dSdksai = [0] * N
        d2Sdksai = [0] * N
        dVdita = [0] * M
        d2Vdita = [0] * M
        for i in range(1, N-1):
                dSdksai[i]      = (S[i+1] - S[i-1]) / (2 * delta_ksai) 
                d2Sdksai[i]     = (S[i-1] - 2*S[i] + S[i+1]) / (delta_ksai**2)  
        for j in range(1, M-1):
                dVdita[j]       = (v[j+1] - v[j-1]) / (2 * delta_ita)  
                d2Vdita[j]      = (v[j-1] - 2*v[j] + v[j+1]) / (delta_ita**2)  
        #For below a, b, c, d, e, f to use            
        
        a, b, c, d, e, f = np.zeros((N, M)), np.zeros((N, M)), np.zeros((N, M)), np.zeros((N, M)), np.zeros((N, M)), np.zeros((N, M))
        for i in range(N - 1):
                for j in range(M - 1):
                        a[i,j] = delta_tau * rho * sigma * v[j] * S[i] / (4 * delta_ksai * delta_tau * dSdksai[i] * dVdita[j])

                        b[i,j] = sigma**2 * delta_tau * v[j] / (2 * delta_ita**2 * dVdita[j]**2 ) - delta_tau * \
                                (kappa * (theta - v[j]) / dVdita[j] - 0.5 * sigma ** 2 * v[j] * d2Vdita[j] / dVdita[j] ** 3 ) / (2 * delta_ita)

                        c[i,j] = delta_tau * v[j] * S[i]**2 / (delta_ksai**2 * dSdksai[i]**2 ) - delta_tau * \
                                ((r - q) * S[i] / dSdksai[i] - 0.5 * v[j] * S[i]**2 * d2Sdksai[i] / dSdksai[i] ** 3 ) / (2 * delta_ksai)

                        d[i,j] = 1 + r * delta_tau + delta_tau * v[j] * S[i]**2 / (delta_ksai**2 * dSdksai[i]**2 ) \
                                + sigma**2 * delta_tau * v[j] / (delta_ita**2 * dVdita[j]**2 )

                        e[i,j] = delta_tau * v[j] * S[i]**2 / (delta_ksai**2 * dSdksai[i]**2 ) + delta_tau * \
                                ((r - q) * S[i] / dSdksai[i] - 0.5 * v[j] * S[i]**2 * d2Sdksai[i] / dSdksai[i] ** 3 ) / (2 * delta_ksai)

                        f[i,j] = sigma**2 * delta_tau * v[j] / (2 * delta_ita**2 * dVdita[j]**2 ) + delta_tau * \
                                (kappa * (theta - v[j]) / dVdita[j] - 0.5 * sigma ** 2 * v[j] * d2Vdita[j] / dVdita[j] ** 3 ) / (2 * delta_ita)
        #These a, b, c, d, e, f are to be plugged in the final equation to get U

        U = U_dic
        for k in range(1, L + 1):
                for i in range(N):
                        for j in range(M):
                                U[k][i,j] = -a[i,j] * U[k+1][i-1,j-1] - b[i,j]*U[k+1][i,j-1] + a[i,j]*U[k+1][i+1,j-1]\
                        - c[i,j]*U[k+1][i-1,j] + d[i,j]*U[k+1][i,j] - e[i,j]*U[k+1][i+1,j] \
                        + a[i,j]*U[k+1][i-1,j+1] - f[i,j]*U[k+1][i,j+1] - a[i,j]*U[k+1][i+1,j+1]
        return U
p_heston = {'kappa':1.50000
           ,'theta':0.04000
           ,'sigma':0.30000
           ,'v0':0.05412
           ,'rho':-0.90000
           ,'lambda':0.00000}
    # Option features
p_option = {'Strike':95
           ,'r':0.02
           ,'q':0.05}
    # Grid parameters
p_grid   = {'NS':30
           ,'NV':30
           ,'NT':1500
           ,'Smin':0
           ,'Smax':2 * p_option['Strike']
           ,'Vmin':0
           ,'Vmax':0.5
           ,'Tmin':0
           ,'Tmax':0.15}    
    
Si = 101.52000
vi = 000.05412
A_Matrix(p_heston, p_option, p_grid)


