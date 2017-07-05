import numpy as np
import matplotlib.pyplot as plt
from math import *
import csv, sys

from skimage import exposure
from skimage import transform
from skimage import color
from skimage import feature
from skimage import segmentation
from scipy import stats

def resizeToArea(im, goal=200000):
    x, y = im.shape[0], im.shape[1]
    r = x / y
    ny = int(sqrt(goal / r))
    nx = int(r * ny)
    return transform.resize(im, (nx, ny, 3), mode='constant')

"""
    Features extraction taken from
        Affective Image Classification using Features
        Inspired byPsychology and Art Theory, Machajdik
"""

def imageToFeatures(im):
    rim = resizeToArea(im)
    cim = color.rgb2hsv(rim) # TODO do not normalize saturation
    patches = segmentation.felzenszwalb(cim)
    # TODO waterfall algorithm to segment the image

    colorF = colorFeatures(cim, rim)
    textureF = textureFeatures(cim)
    compositionF = compositionFeatures(rim)
    print(".", end="")
    sys.stdout.flush();
    return np.concatenate((colorF, textureF, compositionF))

################################################################
def colorFeatures(cim, rim):
    saturation = cim[:,:,1].mean()
    brightness = cim[:,:,2].mean()
    pad = PAD(saturation, brightness)
    hue = hueStatistics(cim)
    # TODO Colorfulness
    colorHisto = colorNames(rim)
    # TODO Itten
    return np.concatenate(([saturation, brightness], \
            pad, hue, colorHisto))

"""
    Pleasure / Arousal / Dominance
"""
def PAD(S, Y):
    p = 0.69 * Y + 0.22 * S
    a = -0.31 * Y + 0.60 * S
    d = 0.76 * Y + 0.32 * S
    return [p, a, d]

def hueStatistics(im):
    #Without weight
    m = stats.circmean(im[:,:,0])
    s = stats.circstd(im[:,:,0]) #std deviation instead of dispersion
    #With weight
    w = np.multiply(im[:,:,0], im[:,:,1])
    wm = stats.circmean(w)
    ws = stats.circstd(w)
    return [m, s, wm, ws]

def colorNames(im):
    index_im = (np.floor(im[:,:,0] / 8) \
            + 32 * np.floor(im[:,:,1] / 8) \
            + 32 * 32 * np.floor(im[:,:,2] / 8)).flatten()
    colors = colorData[index_im.astype(int)]
    histogram = np.bincount(colors, minlength=11)
    return histogram

def getColorData(filename="w2c.txt"):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='\'')
        r = np.zeros(32 * 32 * 32)
        rNb = 0
        for row in reader:
            r[rNb] = np.argmax([float(e) for e in row])
            rNb += 1
    return r.astype(int)

colorData = getColorData()

################################################################

def textureFeatures(cim):
    return np.array([])

################################################################

def compositionFeatures(rim):
    return dynamics(rim)

def showDynamics(grey, lines):
    #Show detected lines
    plt.imshow(grey)
    for (p1, p2) in lines:
        plt.plot((p1[0], p2[0]), (p1[1], p2[1]), '-r')
    plt.xlim((0, grey.shape[1]))
    plt.ylim((grey.shape[0], 0))
    plt.show()

def dynamics(rim):
    grey = color.rgb2grey(rim)
    edges = feature.canny(grey, sigma=3.5)
    lines = transform.probabilistic_hough_line(edges)
    #showDynamics(grey, lines)
    nstatic, ndynamic = 0,0 
    lstatic, ldynamic = 0,0
    for (p1, p2) in lines:
        v = (p2[0] - p1[0], p2[1] - p1[1])
        angle = 180 / np.pi * np.arctan2(v[1], v[0])
        length = np.sqrt(v[0] ** 2 + v[1] ** 2)
        if (angle > -15 and angle < 15) or (angle > 75 and angle < 105):
            #Static line
            nstatic += 1
            lstatic += length
        else:
            #Dynamic line
            ndynamic += 1
            ldynamic += length
    tot = len(lines)
    if tot == 0:
        return np.zeros(6)
    else:
        return np.array([nstatic, ndynamic, nstatic / tot, ndynamic / tot,
        lstatic, ldynamic])
