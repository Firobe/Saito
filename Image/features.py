import numpy as np
import matplotlib.pyplot as plt
from math import *
import csv, sys

from skimage import exposure
from skimage import transform
from skimage import color
from skimage import feature
from skimage import morphology 
from skimage import filters
from scipy import stats, ndimage
from igraph import *

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
    segments = segmentation(cim[:,:,0])
    #print(segments)

    colorF = colorFeatures(cim, rim)
    textureF = textureFeatures(cim)
    compositionF = compositionFeatures(rim)
    print(".", end = "")
    sys.stdout.flush();
    return np.concatenate((colorF, textureF, compositionF))

################################################## SEGMENTATION

"""
    Segmentation functions, based on
        http://cmm.ensmp.fr/~beucher/publi/marcotegui_waterfalls_ismm05.pdf
        Fast implementation of waterfall based on graphs
        Marcotegui, Beucher
"""
def segmentation(im):
    # Watershed
    fim = filters.rank.median(im, morphology.disk(2))
    gradient = filters.rank.gradient(fim, morphology.disk(2))
    basins = watershed(fim, gradient)
    areas, labels = exposure.histogram(basins)
    minimas = [int(fim[basins == v].min()) for v in labels]
    graph = graphFromBasins(basins, labels, minimas)
    spanning = graph.spanning_tree()
    waterfall(spanning)
    #plt.imshow(graph)
    #plt.show()
    showSegmentation(im, basins)

"""
    Performs watershed analysis.
    Returns a matrix of the size of the image, with
    a label corresponding to a basin on each pixel
"""
def watershed(fim, gradient):
    markers = filters.rank.gradient(fim, morphology.disk(5)) < 10
    markers = ndimage.label(markers)[0]
    labels = morphology.watershed(gradient, markers)
    return labels

# To index only the upper part of the adjacency matrix
def gSym(i, j):
    return (j, i) if i > j else (i, j)

"""
    Create an undirected graph from the results
    of the watershed analysis. Each vertex is a basin.
    Vertices are linked by an edge if the corresponding basins
    are adjacent, and the value of the edge of the absolute difference
    between the minimum values of each basin.
"""
def graphFromBasins(basins, labels, minimas):
    L = len(labels)
    n, m = basins.shape
    graph = np.zeros((L, L))
    for i in range(n):
        for j in range(m):
            curVal = basins[i,j] - 1
            if j + 1 < m:
                ind = gSym(curVal, basins[i, j + 1] - 1) #E
                if graph[ind] == 0:
                    graph[ind] = abs(minimas[ind[0]] - minimas[ind[1]])
            if i + 1 < n:
                ind = gSym(curVal, basins[i + 1, j]  - 1) #S
                if graph[ind] == 0:
                    graph[ind] = abs(minimas[ind[0]] - minimas[ind[1]])
            if i + 1 < n and j + 1 < m:
                ind = gSym(curVal, basins[i + 1, j + 1] - 1) #SE
                if graph[ind] == 0:
                    graph[ind] = abs(minimas[ind[0]] - minimas[ind[1]])
            if i + 1 < n and j - 1 >= 0:
                ind = gSym(curVal, basins[i + 1, j - 1] - 1) #SW
                if graph[ind] == 0:
                    graph[ind] = abs(minimas[ind[0]] - minimas[ind[1]])
    
    return Graph.Weighted_Adjacency(graph.tolist(), mode = ADJ_UPPER,
            attr = "weight", loops = False)

"""
    Apply the waterfall algorithm described in the article
"""
def waterfall(spanning):
    mins = Graph()
    # Construct a graph where vertices are minimal edges
    # (with the original edge as "edge" attribute)
    # Two vertices are linked by an edge only if
    # the original two edges have a vertex in common
    # and have the same valuation
    for E in spanning.es:
        if not checkMinimum(spanning, E):
            v = mins.add_vertices(1)
            mins.vs[mins.vcount() - 1]["edge"] = E
    for v1 in mins.vs:
        for v2 in mins.vs:
            if v1 != v2 \
                and v1["edge"]["weight"] == v2["edge"]["weight"] \
                and areIncident(v1["edge"], v2["edge"]):
                mins.add_edges([(v1, v2)])

    # Get the connected components of the constructed graph
    # This should yield the new basins. Each basin (every vertex of
    # the basin) is affected a label in the original graph
    CS = mins.components(mode = WEAK)
    curLabel = 0
    for C in CS:
        for v in C:
            spanning.vs[mins.vs[v]["edge"].source]["label"] = curLabel
            spanning.vs[mins.vs[v]["edge"].target]["label"] = curLabel
        curLabel += 1

    # Some areas are left unlabelled in the original graph
    # Propagate the labels to unlabelled areas the finish the partition
    propagate(spanning)
    print("OALAALAL3")
    for v in spanning.vs:
        print(v["label"])

# Check if two edges have a vertex in common
def areIncident(e1, e2):
    return (e1.source == e2.source or
            e1.source == e2.target or
            e1.target == e2.source or
            e1.target == e2.target)


"""
    Propagate the labels on the spanning tree
    so that the entire tree is labelled at the end
"""
def propagate(g):
    ses = sorted(g.es, key = lambda v: v["weight"])
    for e in ses:
        s, t = g.vs[e.source], g.vs[e.target]
        if s["label"] != t["label"]: #XOR
            if s["label"]: t["label"] = s["label"]
            else: s["label"] = t["label"]
    # TOUT N'EST PAS LABELLED
    # ESSAYER DE DESSINER LE GRAPHE AVANT

"""
    Check if an edge is surrounded only by edges with superior valuation
"""
def checkMinimum(spanning, E):
    curW = E["weight"]
    v1, v2 = E.source, E.target
    for EIind in np.unique(spanning.incident(v1)+spanning.incident(v2)):
        EI = spanning.es[EIind]
        if EI["weight"] < curW: #Maybe <=
            return False
    return True

"""
    Display the segmentation of an image, with a color attributed to each
    label. Labels must be an image with a label on each pixel.
"""
def showSegmentation(image, labels):
    plt.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
    plt.imshow(labels, cmap=plt.cm.spectral, interpolation='nearest', alpha=.7)
    plt.title("Segmented")
    plt.show()

################################################## COLOR FEATURES
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

################################################ TEXTURE FEATURES

def textureFeatures(cim):
    return np.array([])

################################################ COMPOS. FEATURES

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
