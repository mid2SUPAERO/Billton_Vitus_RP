# -*- coding: utf-8 -*-
"""
Created on Sat May 30 19:11:19 2020

@author: billton
"""
import numpy as np
import scipy as sp
import math
import time
import matplotlib.pyplot as plt
from smt.sampling import LHS


#---------------------------Initialisation-----------------------------
Nx = 61 #number of nodes i X
Ny = 61 #number of nodes in Y
Lx = 1.  #Length in X 
Ly = 1.  #Length in Y
dx = Lx/(Nx-1); #Step size in X
dy = Ly/(Ny-1); #Step size in Y
x = np.linspace(0,Lx,Nx)  #Row vector for x
y = np.linspace(0,Ly,Ny)  #Row vector for y
Ti = 300
Tleft = 950

pmin = [0 , 0.0005 , 0.4]
pmax = [0.5 , 0.025 , 0.6]

limits = np.array([[0,0.5], [0.005,0.025], [0.4,0.6]])
sampling=LHS(xlimits=limits)
num = 75
pcomp = sampling(num)
pcomp = np.transpose(pcomp)

wcomp = np.zeros(((Nx-2)*(Ny-2), num))

#-----------------------Functions-------------------------------------
#check indexing
# Computing A1
def computeA1():
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
    
    D2y = np.zeros((Ny-2,Ny-2))
    for i in range (0,Ny-2):
        for j in range (0,Ny-2):
            if i == j:
                D2y[i,j] = -2
            elif j == i+1:
                D2y[i,j] = 1
            elif j == i-1:
                D2y[i,j] = 1
    D2y[Ny-3,Ny-3] = -1
    D2y[0,0] = -1    
    #pdb.set_trace()    
    D2y = (1/dy**2)*D2y
    
    A1 = sp.sparse.kron(D2x, np.eye(Ny-2)) + sp.sparse.kron(np.eye(Nx-2), D2y)
    
    return A1

#computing A2
def computeA2():    
    D1x = np.zeros((Nx-2,Nx-2))
    for i in range (0,Nx-2):
        for j in range (0,Nx-2):
            if i == j:
                D1x[i,j] = 1
            elif j == i+1:
                D1x[i,j] = 0
            elif j == i-1:
                D1x[i,j] = -1   
        
    D1x = (1/dx)*D1x
    
    A2 = np.kron(D1x, np.eye(Ny-2))
    
    return A2

#computing B1
def computeB1(u, kappa, ybar):
    TleftBC = np.zeros((Ny-2,1))
    b1 = np.zeros((Ny-2,1))
    for i in range(1,Ny-1):  #adjust range from matlab
        if y[i]<Ly/3:
            TleftBC[i-1,0] = Ti
            b1[i-1,0] = -((u/dx) + kappa/dx**2)*TleftBC[i-1,0]
        elif y[i] > 2*Ly/3:
            TleftBC[i-1,0] = Ti
            b1[i-1,0] = -((u/dx) + kappa/dx**2)*TleftBC[i-1,0]
        else:
            TleftBC[i-1,0] = 300 + 325*(np.sin(3*math.pi*np.abs(y[i]-ybar))+1)
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

    
#reference solution
tic2 = time.time()
for i in range (0,num):
    [A,b,TleftBC] = buildadvdefop(pcomp[0,i], pcomp[1,i], pcomp[2,i])
    wcomp[:,i] = np.linalg.solve(A,b).T.squeeze()
tfull = time.time() - tic2
print(f'Time for full snapshot creation is {tfull:.3f} sec ')

#Full Soln
# temp = np.transpose(np.reshape(wcomp[:,1],(59,59)))
# temp = np.hstack((TleftBC,temp))
# x1 = np.linspace(0,1,60)
# y1 = np.linspace(0,1,59)
# X,Y = np.meshgrid(x1,y1)
# plt.contourf(X,Y,wcomp, temp = 200)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.colorbar()
# plt.show()

#np.save('Param.npy',pcomp)
#np.save('Snapshot.npy',wcomp)