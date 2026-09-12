
import sys
import numpy as np
sys.path.insert(0, r"D:/Projects/flylingo/brain")
from brain.learning.adapter import _softmax

def H(z):
    p = _softmax(z)
    return float(-(p*np.log(p+1e-9)).sum())

def analytic_true(z):
    p = _softmax(z)
    h = -(p*np.log(p+1e-9)).sum()
    return -p*(h + np.log(p+1e-9))

def analytic_coded(z):
    p = _softmax(z)
    h = -(p*np.log(p+1e-9)).sum()
    return p*(h - np.log(p+1e-9))

def fd(z, eps=1e-6):
    g = np.zeros_like(z)
    for j in range(len(z)):
        zp = z.copy(); zp[j] += eps
        zm = z.copy(); zm[j] -= eps
        g[j] = (H(zp) - H(zm)) / (2*eps)
    return g

for z in [np.array([0.,0.]),
          np.array([0.5,-0.3,0.9,0.1]),
          np.array([2.,-1.,0.3,-2.5]),
          np.array([0.,0.,0.,0.])]:
    f = fd(z); t = analytic_true(z); c = analytic_coded(z)
    print("z=", np.round(z,3))
    print("  finite-difference :", np.round(f,6), " sum %.2e" % f.sum())
    print("  analytic TRUE     :", np.round(t,6), " max|err| vs FD %.3e" % np.abs(f-t).max())
    print("  analytic AS-CODED :", np.round(c,6), " max|err| vs FD %.3e  sum %.4f" % (np.abs(f-c).max(), c.sum()))
    print("  as-coded is -true? max|err| %.3e" % np.abs(t+c).max())
