import numpy as np
import scipy as sp
import math
import time

from ezyrb import POD, RBF, Database
from ezyrb import ReducedOrderModel as ROM
#%matplotlib inline

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

npoints = 10
ncomp = npoints**3
wcomp = np.zeros(((Nx-2)*(Ny-2), ncomp))


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


#POD offline 
snapshots = np.load('D:\Actual Purpose\Supaero\Research Project\Codes\Snapshots\Snapshot75b.npy')
print(f'Total number of columns in snapshot matrix = {len(snapshots[0])}')
limit = int(0.8*len(snapshots[0]))
snapshots = snapshots[:,0:limit]
snapshots = np.transpose(snapshots)    

param = np.load('D:\Actual Purpose\Supaero\Research Project\Codes\Param\Param75b.npy')
param_snap = param[:,0:limit]
param_snap = np.transpose(param_snap)

db = Database(param_snap, snapshots)
pod = POD('svd')
rbf = RBF()
rom = ROM(db, pod, rbf)
rom.fit();

#POD Online
param_train = np.transpose(param[:,limit:len(param[0])])
Error_array = np.zeros((len(param[0])-limit,1))
train_limit = len(param[0])-limit

for i in range(0,train_limit):
    param_i = param_train[i,:]
    [A,b,TleftBC] = buildadvdefop(param_i[0], param_i[1], param_i[2])
    wcomp = np.linalg.solve(A,b).T.squeeze()
    pred_sol = rom.predict(param_i)
    MSE = np.square(np.subtract(wcomp,pred_sol)).mean()
    RMSE = math.sqrt(MSE)
    Error_array[i,:] = RMSE

#Error
training_error = np.average(Error_array)
print(f"RMS error is {training_error:.4f}")