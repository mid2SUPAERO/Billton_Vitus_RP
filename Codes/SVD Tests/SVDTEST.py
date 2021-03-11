# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 01:55:01 2020

@author: billt
"""

import numpy as np
import scipy as sp
import time
#from tensorflow.compat.v1 import enable_eager_execution
import tensorflow as tf
import torch
from tqdm import tqdm, trange
from sklearn.decomposition import TruncatedSVD
#tf.enable_eager_execution()

A = np.random.rand(100000,150)

time1 = []
time2 = []
time3 = []
time4 = []
time5 = []
time6 = []
time7 = []

# for i in trange(10):

print("\n starting Tensorflow (with S U V)")
tic1 = time.time()
s1, u1, v1 = tf.linalg.svd(A)
#s = tf.linalg.svd(A, compute_uv=False)
time1.append(time.time() - tic1)


print("\n starting Tensorflow (Only S)")
tic5 = time.time()
s5 = tf.linalg.svd(A, compute_uv=False)
#s = tf.linalg.svd(A, compute_uv=False)
time5.append(time.time() - tic5)


print("\n starting Pytorch")
device = torch.device("cuda")
torch.set_default_tensor_type('torch.cuda.FloatTensor')
B = torch.tensor(A).to(device)
tic2 = time.time()
s2,u2,v2 = torch.svd(B)
time2.append(time.time() - tic2)
   

# print("\n starting numpy")
# tic3 = time.time()
# s3,u3,v3 = np.linalg.svd(A)
# time3.append(time.time() - tic3)


# print("\n starting scipy")
# tic4 = time.time()
# s4,u4,v4 = sp.linalg.svd(A)
# time4.append(time.time() - tic4)

tic6 = time.time()
svd = TruncatedSVD(n_components=50)
svd.fit(A)
time6.append(time.time()-tic6)

tic7 = time.time()
u7,s7,v7 = sp.sparse.linalg.svds(A,k=50)
time7.append(time.time()-tic7)

print("\n Tensorflow completed in ",np.mean(time1), "secs")
print("Tensorflow (Only S) completed in ",np.mean(time5), "secs")
print("Pytorch completed ",np.mean(time2),"secs")
print("Numpy completed in ",np.mean(time3),"secs")
print("Scipy completed in ",np.mean(time4),"secs")
print("Truncated SVD (randomized) completed in ",time6, "secs")
print("Scipy Sparse SVD completed in ",time7, "secs")