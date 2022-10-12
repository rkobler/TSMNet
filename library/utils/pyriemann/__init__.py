import numpy as np
from scipy.linalg import eigvalsh
from pyriemann.tangentspace import TangentSpace

def squared_airm(A, B):
    return np.square(np.log(eigvalsh(A, B))).sum()

def airm(A,B):
    return np.sqrt(squared_airm(A,B))

def geom_mean(As):
    ts = TangentSpace()
    ts.fit(As)
    return ts.reference_

def tsm(As):
    ts = TangentSpace()
    ts.fit(As)
    return ts.transform(As)