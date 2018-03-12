import numpy as np
import tensorflow as tf
import h5py

A01T = h5py.File("../project_datasets/A01T_slice.mat", 'r')
X = np.copy(A01T["image"])
y = np.copy(A01T["type"])
y = y[0, 0:X.shape[0]:1]
y = np.asarray(y, dtype=np.int32)

# np.savetxt('test.txt', X, fmt='%5s', delimiter=',')

for i in range(len(y)):
    if y[i] == 769:
        y[i] = 0
    elif y[i] == 770:
        y[i] = 1
    elif y[i] == 771:
        y[i] = 2
    elif y[i] == 772:
        y[i] = 3
    else:
        y[i] = 0
X = X.reshape(-1, 25*1000)
y = y[~np.isnan(X).any(axis=1)]
X = X[~np.isnan(X).any(axis=1)]
X = np.copy(X.reshape(-1, 25, 1000))