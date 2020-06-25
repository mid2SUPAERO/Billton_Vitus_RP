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
import tensorflow as tf
from scipy.sparse import linalg
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt


#---------------------------Initialisation-----------------------------
Nx = 61 #number of nodes i X
Ny = 61 #number of nodes in Y
Lx = 1  #Length in X 
Ly = 1  #Length in Y
dx = Lx/(Nx-1); #Step size in X
dy = Ly/(Ny-1); #Step size in Y
x = np.linspace(0,Lx,Nx-1)  #Row vector for x
y = np.linspace(0,Ly,Ny-1)  #Row vector for y
Ti = 300
Tleft = 950

pmin = [0 , 0.0005 , 0.4]
pmax = [0.5 , 0.025 , 0.5]

pminarray = np.transpose(np.array([[0 , 0.0005 , 0.4]]))
pmaxarray = np.transpose(np.array([[0.5 , 0.025 , 0.5]]))


npoints = 10
ucomp = np.linspace(pmin[0], pmax[0], npoints)
kappacomp = np.linspace(pmin[1], pmin[1], npoints)
ybarcomp = np.linspace(pmin[2], pmax[2], npoints)
ncomp = npoints**3
wcomp = np.zeros(((Nx-2)*(Ny-2), ncomp))
pcomp = np.zeros((3,ncomp))


#-----------------------Functions-------------------------------------
#check indexing
# Computing A1
def computeA1():
    # import pdb; pdb.set_trace()
    # datax = [np.ones((Nx-2)), -2*np.ones((Nx-2)), np.ones((Nx-2))]
    # datax[1][Nx-3] = -1
    # D2x = sp.sparse.spdiags(datax,[-1, 0, 1], Nx-2, Nx-2)
    # #D2x[Nx-3, Nx-3] = -1
    
    D2x = np.zeros((Nx-2,Nx-2))
    for i in range (0,Nx-2):
        for j in range (0,Nx-2):
            if i == j:
                D2x[i,j] = -2
            elif j == i+1:
                D2x[i,j] = 1
            elif j == i-1:
                D2x[i,j] = 1
    D2x[Nx-3,Nx-3] = -1
    D2x = (1/dx**2)*D2x
    
    # datay = [np.ones(Ny-2), -2*np.ones(Ny-2), np.ones(Ny-2)]
    # datay[1][Ny-3] = -1
    # datay[1][0] = -1
    # D2y = sp.sparse.spdiags(datay,[-1, 0, 1], Ny-2, Ny-2)
    # #D2y[Ny-3, Ny-3] = -1
    # #D2y[0,0] = -1
    
    D2y = np.zeros((Ny-2,Ny-2))
    for i in range (0,Ny-2):
        for j in range (0,Ny-2):
            if i == j:
                D2y[i,j] = -2
            elif j == i+1:
                D2y[i,j] = 1
            elif j == i-1:
                D2y[i,j] = 1
    D2x[Ny-3,Ny-3] = -1    
    
    D2y = (1/dy**2)*D2y
    
    A1 = np.kron(D2x, np.eye(Ny-2)) + np.kron(np.eye(Nx-2), D2y)
    
    return A1

#computing A2
def computeA2():    
    #data1x = [-np.ones(Nx-2), np.ones(Nx-2)]
    #D1x = sp.sparse.spdiags(data1x,[-1,0], Nx-2, Nx-2, format=None)
    D1x = np.zeros((Nx-2,Nx-2))
    for i in range (0,Nx-2):
        for j in range (0,Nx-2):
            if i == j:
                D1x[i,j] = -2
            elif j == i+1:
                D1x[i,j] = 1
            elif j == i-1:
                D1x[i,j] = 1   
    
    D1x = (1/dx)*D1x
    
    A2 = np.kron(D1x, np.eye(Ny-2))
    
    return A2

#computing B1
def computeB1(u, kappa, ybar):
    #import pdb; pdb.set_trace()
    TleftBC = np.zeros((Ny-2,1))
    b1 = np.zeros((Ny-2,1))
    for i in range(1,Ny-1):  #adjust range from matlab
        if y[i]<Ly/3:
            TleftBC[i-1,0] = Ti
            b1[i-1,0] = -(u/dx + (kappa/dx)**2)*TleftBC[i-1,0]
        elif y[i] > 2*Ly/3:
            TleftBC[i-1,0] = Ti
            b1[i-1,0] = -(u/dx + kappa/dx**2)*TleftBC[i-1,0]
        else:
            TleftBC[i-1,0] = Ti+(Tleft-Ti)*(np.sin(np.pi*abs(y[i]-ybar)/(1/3))+1)/2
            b1[i-1,0] = -(u/dx + kappa/dx**2)*TleftBC[i-1,0]
    return b1, TleftBC


#Advection Diffusion Operator
def buildadvdefop(u, kappa, ybar):
    if u<0:
        print('u must be positive')
    [b1, TleftBC] = computeB1(u, kappa, ybar)
    b = np.vstack((b1, np.zeros(((Nx-3)*(Ny-2),1))))
    
    A1 = computeA1()
    A2 = computeA2()
    A = (kappa*A1) - (u*A2)
    return A, b, TleftBC

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
    return Ar1, Ar2, Ar3, Ar4, Ar5, Z1, Z2, V1             
                      
#POD Function 
def POD(ntest, wcomp, pcomp, ncomp, pminarray, pmaxarray, npoints, Id):
    emax = 0
    eavg = 0
    temps = np.zeros((ntest,1))
    for itest in range(0,ntest):
        tic = time.time()
        nsamples = 10 
        niterations = nsamples-1
        ncandidates = 1000
        pcandidate = pminarray*np.ones((1,ncandidates)) + np.multiply(np.transpose(lhs(3,ncandidates)),((pmaxarray-pminarray)*np.ones((1,ncandidates))))
        
        #sample 1 point in the middle
        pmiddle = (pminarray+pmaxarray)/2
        wsamples = np.zeros(((Nx-2)*(Nx-2),nsamples))
        [A,b,_] = buildadvdefop(pmiddle[0,0], pmiddle[1,0], pmiddle[2,0])
        wmiddle = sp.linalg.solve(A,b)
        wsamples[:,0] = wmiddle.T.squeeze()
        [v,_,_] = np.linalg.svd(wsamples)
        
        for iiteration in range (0,niterations):
            [Ar1, Ar2, Ar3, Ar4, Ar5, Z1, Z2, V1] = buildprecompops(v)
            maxerror = 0
            candidatemaxerror = 0
            
            for icandidate in range (0,ncandidates):
                [b1,_] = computeB1(pcandidate[0,icandidate-1], pcandidate[1,icandidate-1], pcandidate[2,icandidate-1] )
                Ar = pcandidate[1, icandidate-1]*Ar4 - pcandidate[0,icandidate-1]*Ar5
                br = np.transpose(V1)@b1
                
                q = sp.linalg.solve(Ar, br)
                q = np.transpose(q)
                
                errorindicator = np.transpose(q)@((((pcandidate[1,icandidate-1])**2)*Ar1) - pcandidate[1,icandidate-1]*pcandidate[0,icandidate-1]*Ar2 + pcandidate[0,icandidate-1]**2*Ar3)@q - 2*(pcandidate[1,icandidate-1]*np.transpose(b1)@Z1 - pcandidate[0,icandidate-1]*np.transpose(b1)@Z2)@q + np.transpose(b1)@b1
                if icandidate == 0 or maxerror < errorindicator:
                    candidatemaxerror = icandidate
                    maxerror = errorindicator
            
            [A,b,_] = buildadvdefop(pcandidate[0,candidatemaxerror-1],pcandidate[1,candidatemaxerror-1], pcandidate[2,candidatemaxerror-1])
            wsamples[:,iiteration+1] = sp.linalg.solve(A,b)
            [v,_,_] = np.linalg.svd(wsamples[:,0:iiteration+1],0)
            
        k = np.ndarray.size(v,2)
        q = np.zeros(k,ncomp)
        error = np.zeros(ncomp,1)
        count = 0
        for ip in range (0, npoints):
            for jp in range (0, npoints):
                for kp in range (0, npoints):
                    count = count + 1
                    [A,b,_] = buildadvdefop(pcomp[0,count-1], pcomp[1,count-1], pcomp[2,count-1])
                    Ar = np.transpose(v)*A*v
                    br = np.transpose(v)*b
                    
                    q[:,count] = sp.lianalg.solve(Ar,br)
                    wtilde = v*q[:,count]
                    error[count,1] = np.linalg.norm(wcomp[:,count-1]-wtilde,2)/np.linalg.norm(wcomp[:,count-1],2)
                    print(count)
                    if count == id:
                        w = wtilde
        emax = emax + 1/ntest*max(error)
        eavg = eavg + 1/ntest*np.mean(error)
        temps[itest] = time.time() - tic
        
    return emax, eavg, temps, w, v
    
#reference solution
count = 0
tic2 = time.time()
for ip in range (0,npoints):
    for jp in range (0,npoints):
        for kp in range (0,npoints):
            count = count + 1
            pcomp[:,count-1] = [ucomp[ip], kappacomp[jp], ybarcomp[kp]]
            [A,b,TleftBC] = buildadvdefop(pcomp[0,count-1], pcomp[1,count-1], pcomp[2,count-1])
            wcomp[:,count-1] = sp.linalg.solve(A,b).T.squeeze()
tfull = time.time() - tic2
print('Time for full case is %f', tfull)

#Plot reference solution[


#POD Solution    
ntest = 20
[emax, eavg, temps, w, v] = POD(ntest, wcomp, pcomp, ncomp, pmin, pmax, npoints, 291)
print('Errors observed are %f (avg) and %f (max)', eavg, emax)
pod_t_avg = np.mean(temps)   
    
    
    
    