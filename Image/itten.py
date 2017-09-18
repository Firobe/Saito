import numpy as np
from skimage import exposure

# Itten colors features
def itten(cim, segments):
    s, n = segments[0], segments[1]
    size, labels = exposure.histogram(s)
    n = len(size)
    #Delete empty bins
    toDel = [i for i in range(n) if size[i] == 0]
    size = [size[i] for i in range(n) if not (i in toDel)]
    labels = [labels[i] for i in range(n) if not (i in toDel)]

    # Average H, S, B of each segment
    hues = np.array([cim[s == i][:,0].mean() * 360 for i in labels])
    sSegs = np.array([cim[s == i][:,1].mean() for i in labels])
    bSegs = np.array([cim[s == i][:,2].mean() for i in labels])

    # Translate to itten model
    ittenLight = lightMembership(bSegs)
    ittenSat = satMembership(sSegs)

    # TODO Max ?
    lightDarkC = standardContrast(ittenLight, size)
    saturationC = standardContrast(ittenSat, size)
    # TODO Contrast of hues (vector-based measure of hue spread ??)
    complementCA, complementCM = complementsContrast(hues)
    warmColdCA, warmColdCM = warmColdContrast(hues)
    return np.array([lightDarkC, saturationC, complementCA,
        complementCM, warmColdCA, warmColdCM])

"""
    Image Retrieval by Emotional Semantics:
        A Study of Emotional Space and Feature Extraction,
    Wang et al.
    http://ieeexplore.ieee.org/xpls/icp.jsp?arnumber=4274431
"""
# Label each value of x (in [0,1]) with a label in [[0,5]]
# (from very dark to very light)
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
        # Edge case management (TODO improve it so it's linear in time)
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

# Label each value of x (in [0,1]) with 0, 1, 2
# (low sat, middle sat, high sat)
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
def interv(c, x):
    for j in range(1, len(c) - 2):
        if c[j] < x and x <= c[j + 1]:
            return j
    raise NameError("Invalid X value")

# Weighted standard deviation on values
def standardContrast(values, weights):
    av = np.average(values, weights = weights)
    var = np.average((values - av) ** 2, weights = weights)
    return np.sqrt(var)

# Returns average and maximal difference of hue
def complementsContrast(hues):
    n = len(hues)
    if n == 1: return (0, 0)
    S = 0
    M = 0
    for i in range(n):
        for j in range(i + 1, n):
            ad = abs(hues[i] - hues[j])
            d = min(ad, 360 - ad)
            S += d
            if d > M: M = d
    return ((2 * S) / (n ** 2 - n), M)

def warmColdContrast(hues):
    n = len(hues)
    if n == 1: return (0, 0)
    S, M = 0, 0
    temps = [{}] * n
    for i in range(n):
        h = hues[i]
        warm = np.cos(h - 50) if (0 <= h and h < 140) or (320 <= h
                and h <= 360) else 0
        cold = np.cos(h - 230) if 140 <= h and h < 320 else 0
        neutral = 1 - (warm + cold)
        temps[i] = {'cold': cold, 'warm': warm, 'neutral': neutral}

    for i in range(n):
        for j in range(i + 1, n):
            s1 = temps[i]['warm'] * temps[j]['warm'] +\
                    temps[i]['cold'] * temps[j]['cold'] +\
                    temps[i]['neutral'] * temps[j]['neutral']
            s2x = lambda x: temps[x]['warm'] ** 2 + temps[x]['cold'] ** 2 +\
                    temps[x]['neutral'] ** 2
            s2 = np.sqrt(s2x(i) * s2x(j))
            contrast = s1 / s2
            S += contrast
            if contrast > M: M = contrast
    average = (2 * S) / (n ** 2 - n)
    return (average, M)
