# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 12:13:08 2022

@author: chunl
"""

import numpy as np
import math
from scipy.optimize import minimize

# heston model
k = 100; 
rou = 0.5
T = 0.5; N = 3000; M = 3500; L = 9000
sigma = 0.2 # sigma is the volatility of volatility and it is constant in this situation
smin = 0; smax = k*2
vmax = 0.8; vmin = 0.1
alpha = N; beta = M #??!!!

ht = T/L # T/N
hs = smax/N # smax/N
hsig = 1/M
c1 = (math.asinh((smax - k)/N)); c2 = (math.asinh((smin - k)/N))
d = math.asinh(vmax/M)
def space(space,alpha,c1,c2):
    # the input is wrt delta space
    s = k+alpha*np.sinh(c1*space+c2*(1-space))
    return s

def vol(vol,beta,d):
    v = beta*np.sinh(d*vol)
    return v

def space_d(space,alpha,c1,c2):
    s = (c1-c2)*alpha*np.cosh(c1*space+c2*(1-space))
    return s

def vol_d(vol,beta,d):
    v = beta*d*np.cosh(d*vol)
    return v
#signal = c1-c2
# c2 is always smaller than c1 and therefore we do not have to consider signal
target = lambda x: -abs(ht*rou*sigma/(4*hs*hsig)*vol(x[1],beta,d)*space(x[0],alpha,c1,c2)/(space_d(x[0],alpha,c1,c2)*vol_d(x[1],beta,d)))

'''
def f(x):
    return -abs(vol(x[1],3500,0.002)*space(x[0],3000,0.03,-0.03)/(space_d(x[0],3000,0.03,-0.03)*vol_d(x[1],3500,0.0002)))
'''

cons = ({'type':'ineq', 'fun':lambda x:x[0]-smin},
        {'type':'ineq', 'fun':lambda x:smax-x[0]},
        {'type':'ineq', 'fun':lambda x:x[1]-vmin},
        {'type':'ineq', 'fun':lambda x:vmax-x[1]})
aa = minimize(target,[0,0.1],constraints =cons).fun
print(abs(aa)<1)

# find the parameters (if needed!!!)
'''
for i in np.arange(1000,40000,200):#N
    for j in np.arange(1000,40000,200):#M
        for z in np.arange(1000,40000,200):#L
            N = i; M = j; L = z
            ht = T/L # T/N
            hs = smax/N # smax/N
            hsig = 1/M
            c1 = (math.asinh((smax - k)/N)); c2 = (math.asinh((smin - k)/N))
            d = math.asinh(vmax/M)
            target = lambda x: -abs(ht*rou*sigma/(4*hs*hsig)*vol(x[1],beta,d)*space(x[0],alpha,c1,c2)/(space_d(x[0],alpha,c1,c2)*vol_d(x[1],beta,d)))
            aa = minimize(target,[0,0.1],constraints =cons).fun
            if (abs(aa)<1) == True:
                print([i,j,z])
                break

'''











