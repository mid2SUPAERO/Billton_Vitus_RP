# -*- coding: utf-8 -*-
"""
Created on Sat May 30 19:11:19 2020

@author: billton
"""
import numpy as np
import scipy as sp
import math
import time
from pyDOE import *

#---------------------------Initialisation-----------------------------
Nx = 61 #number of nodes i X
Ny = 61 #number of nodes in Y
Lx = 1  #Length in X 
Ly = 1  #Length in Y
dx = Lx/(Nx-1); #Step size in X
dy = Ly/(Ny-1); #Step size in Y
x = np.linspace(0,Lx,Nx-1)  #Row vector for x
y = np.arange(0,Ly,Ny-1)  #Row vector for y
Ti = 300
Tleft = 950

pmin = [0 , 0.0005 , 0.4]
pmax = [0.5 , 0.025 , 0.5]


npoints = 10
ucomp = np.linspace(pmin[0], pmax[0], npoints)
kappacomp = np.linspace(pmin[1], pmin[1], npoints)
ybarcomp = np.linspace(pmin[2], pmax[2], npoints)
ncomp = npoints^3
wcomp = np.zeros(((Nx-2)*(Ny-2), ncomp))
pcomp = np.zeros((3,ncomp))


#-----------------------Functions-------------------------------------
# Computing A1
def computeA1():
    D2x = sp.sparse.spdiags([np.ones(Nx-2,1), -2*np.ones(Nx-2,1), np.ones(Nx-2,1)],[-1, 0, 1], Nx-2, Nx-2, format=None)
    D2x[Nx-2, Nx-2] = -1
    D2x = (1/dx^2)*D2x
    
    D2y = sp.sparse.spdiags([np.ones(Ny-2,1), -2*np.ones(Ny-2,1), np.ones(Ny-2,1)],[-1, 0, 1], Ny-2, Ny-2, format=None)
    D2y[Ny-2, Ny-2] = -1
    D2y = (1/dy^2)*D2y
    
    A1 = np.kron(D2x, np.eye(Ny-2)) + np.kron(np.eye(Nx-2), D2y)
    
    return A1

#computing A2
def computeA2():
    D1x = sp.sparse.spdiags([-np.ones(Nx-2,1), np.ones(Nx-2,1)],[-1,0], Nx-2, Nx-2, format=None)
    D1x = (1/dx)*D1x
    
    A2 = np.kron(D1x, np.eye(Ny-2))
    
    return A2

#computing B2
def computeB1(u, kappa, ybar):
    TleftBC = np.zeros(Ny-2,1)
    b1 = np.zeros(Ny-2,1)
    for i in range(1,Ny):  #adjust range from matlab
        if y[i]<Ly/3:
            TleftBC[i-1,1] = Ti
            b1[i-1,1] = -(u/dx + kappa/dx**2)*Tleft[i-1,1]
        elif y[i] > 2*Ly/3:
            TleftBC[i-1,1] = Ti
            b1[i-1,1] = -(u/dx + kappa/dx**2)*Tleft[i-1,1]
        else:
            TleftBC[i-1,1] = Ti+(Tleft-Ti)*(np.sin(np.pi*abs(y[i]-ybar)/(1/3))+1)/2
            b1[i-1,1] = -(u/dx + kappa/dx**2)*Tleft[i-1,1]
    return b1, TleftBC

#Advection Diffusion Operator
def buildadvdefop(u, kappa, ybar):
    if u<0:
        print('u must be positive')
    [b1, TleftBC] = computeB1(u, kappa, ybar)
    b = np.append([b1], [np.zeros(((Nx-2)*(Nx-3),1))])
    
    A1 = computeA1()
    A2 = computeA2()
    A = (kappa*A1) - (u*A2)

#Pre computed Operators   
def buildprecompops(v):
    A2 = computeA2()
    A1 = computeA1()
    
    Y1 = A1*v
    Y2 = A2*v
    
    Z1 = Y1[0:Ny-2,:]
    Z2 = Y2[0:Ny-2,:]
    
    Ar1 = np.transpose(Y1)*Y1
    Ar2 = np.transpose(Y1)*Y2 + np.transpose(Y2)*Y1
    Ar3 = np.transpose(Y2)*Y2
    Ar4 = np.transpose(v)*Y1
    Ar5 = np.transpose(v)*Y2
    
    V1 = v[0:(Ny-2),:]             
                      
#POD Function 
def POD(ntest, wcomp, pcomp, ncomp, pmin, pmax, npoints, Id):
    emax = 0
    eavg = 0
    temps = np.zeroes((ntest,1))
    for itest in range(0,ntest):
        tic = time.time()
        nsamples = 10 
        niterations = nsamples-1
        ncandidates = 1000
        pcandidate = pmin*np.ones((1,ncandidates)) + np.transpose(lhs(ncandidates,3))*((pmax-pmin)*np.ones((1,ncandidates)))
        
        #sample 1 point in the middle
        pmiddle = (pmin+pmax)/2
        wsamples = np.zeros(((Nx-2)*(Nx-2),nsamples))
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    