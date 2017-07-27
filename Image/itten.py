import numpy as np
from skimage import exposure

# Itten colors features
def itten(cim, segments):
    s, n = segments[0], segments[1]
    size, labels = exposure.histogram(s)
    # Average H, S, B of each segment
    hSegs = np.array([cim[s == i][:,0].mean() for i in labels])
    sSegs = np.array([cim[s == i][:,1].mean() for i in labels])
    bSegs = np.array([cim[s == i][:,2].mean() for i in labels])

    # Translate to itten model
    ittenLight = lightMembership(bSegs)
    ittenSat = satMembership(sSegs)
    ittenHues = np.array(
            [11 if h == 1 else np.floor(h * 12) for h in hSegs])
    return []

"""
    Image Retrieval by Emotional Semantics:
        A Study of Emotional Space and Feature Extraction,
    Wang et al.
    http://ieeexplore.ieee.org/xpls/icp.jsp?arnumber=4274431
"""
def lightMembership(x):
    n = len(x)
    # STEP 1
    c = np.zeros(7)
    c[0] = min(x)
    c[6] = max(x)
    for j in range(1, 6):
        c[j] = c[0] + j * (c[6] + c[0]) / 6.

    # STEP 2
    while True:
        backup = c.copy()
        U = np.zeros((n, 5))
        for i in range(n):
            for j in range(1, 6):
                # Rule 1
                if x[i] <= c[1]:
                    r = 1 if j == 1 else 0
                # Rule 2
                elif x[i] > c[5]:
                    r = 0 if j != 5 else 1
                # Rule 3
                else:
                    k = interv(c, x[i])
                    v = (c[k + 1] - x[i]) / (c[k + 1] - c[k]) #!!!!
                    if j == k: r = v
                    elif j == k + 1: r = 1 - v
                    else: r = 0 #!!!!
                U[i, j - 1] = r
        # STEP 3
        for j in range(1, 6):
            # TODO Manage edge cases (0)
            s1 = sum(U[:,j - 1] * x)
            s2 = sum(U[:,j - 1])
            if s1 == 0 or s2 == 0:
                c[j] = -1 # Edge case
            else:
                c[j] = s1 / s2
        # Edge case management (TODO improve it so it's linear)
        for j in range(1, 6):
            if c[j] == -1:
                A, B = j, j
                while c[A] == -1: A-=1
                while c[B] == -1: B+=1
                c[j] = (c[A] + c[B]) / 2
        if (np.abs(backup - c) < 0.01).all():
            break
    R = np.zeros(U.shape[0])
    for i in range(len(R)):
        R[i] = U[i].argmax()
    return R

def satMembership(x):
    n = len(x)
    U = np.zeros((n, 3))
    for i in range(n):
        C = x[i] * 100
        # LS
        if C < 10: LS = 1
        elif C > 27: LS = 0
        else: LS = (27 - C) / 17
        # MS
        if 10 <= C and C <= 27:
            MS = (C - 10) / 17
        elif 27 <= C and C <= 51:
            MS = (51 - C) / 24
        else: MS = 0
        # HS
        if C < 27: HS = 0
        elif C > 51: HS = 1
        else: HS = (C - 27) / 24
        U[i,0], U[i,1], U[i,2] = LS, MS, HS
    R = np.zeros(n)
    for i in range(n):
        R[i] = U[i].argmax()
    return R

# Given c = [c_1, ..., c_n] (sorted) and x
# such that c_1 < x <= c_n
# Returns j such that c_j < x <= c_(j+1)
# TODO binary search
def interv(c, x):
    for j in range(1, len(c) - 2):
        if c[j] < x and x <= c[j + 1]:
            return j
