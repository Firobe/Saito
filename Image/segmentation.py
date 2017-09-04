from igraph import *
import numpy as np
from skimage import morphology 
from skimage import filters
from skimage import img_as_ubyte
from skimage import exposure
from skimage import color
from scipy import stats, ndimage
import matplotlib.pyplot as plt

import warnings

"""
    Fast implementation of waterfall based on graphs,
    Marcotegui, Beucher
    http://cmm.ensmp.fr/~beucher/publi/marcotegui_waterfalls_ismm05.pdf
"""
def segmentation(orim):
    # Watershed
    with warnings.catch_warnings(): # Ignore precision warning
        warnings.simplefilter("ignore")
        im = img_as_ubyte(orim[:,:,0])
    fim = filters.rank.median(im, morphology.disk(2))
    gradient = filters.rank.gradient(fim, morphology.disk(2))
    basins = watershed(fim, gradient)
    areas, labels = exposure.histogram(basins)
    graph = graphFromBasins(basins, labels, im)

    layout = graph.layout("kk")
    #plot(graph, layout=layout)

    spanning = graph.spanning_tree()
    n = waterfall(spanning)
    reduced = simplify(basins, spanning)
    #showSegmentation(orim, gradient, basins, reduced)
    return (reduced, n)
    #return (basins, len(labels))

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

# Update graph (for graphFromBasins) with correct edge valuation
def updateGraph(graph, ind1, ind2, basins, im):
    val1 = basins[ind1] - 1
    val2 = basins[ind2] - 1
    if val1 != val2:
        # TODO Is it really the good value ?
        diff = abs(int(im[ind1]) - int(im[ind2]))
        ind = gSym(val1, val2)
        if diff != 0 and (graph[ind] == 0 or graph[ind] > diff):
            graph[ind] = diff if diff != 0 else 1

"""
    Create an undirected graph from the results
    of the watershed analysis. Each vertex is a basin.
    Vertices are linked by an edge if the corresponding basins
    are adjacent, and the value of the edge of the absolute difference
    between the minimum values of each basin.
"""
def graphFromBasins(basins, labels, im):
    L = len(labels)
    n, m = basins.shape
    graph = np.zeros((L, L))
    for i in range(n):
        for j in range(m):
            if j + 1 < m:
                updateGraph(graph, (i,j), (i, j + 1), basins, im)
            if i + 1 < n:
                updateGraph(graph, (i,j), (i + 1, j), basins, im)
            if i + 1 < n and j + 1 < m:
                updateGraph(graph, (i,j), (i + 1, j + 1), basins, im)
            if i + 1 < n and j - 1 >= 0:
                updateGraph(graph, (i,j), (i + 1, j - 1), basins, im)
    
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
    return curLabel

# Check if two edges have a vertex in common
def areIncident(e1, e2):
    return (e1.source == e2.source or
            e1.source == e2.target or
            e1.target == e2.source or
            e1.target == e2.target)

def hasA(t, a):
    return a in t.attributes() and t[a] != None

"""
    Propagate the labels on the spanning tree
    so that the entire tree is labelled at the end
"""
def propagate(g):
    ses = sorted(g.es, key = lambda v: v["weight"])
    for e in ses:
        s, t = g.vs[e.source], g.vs[e.target]
        if hasA(s, "label") != hasA(t, "label"): #XOR
            if hasA(s, "label"): t["label"] = s["label"]
            else: s["label"] = t["label"]

"""
    Given a graph with a 'label' attribute on each vertex
    Simplify a basins matrix
"""
def simplify(basins, g):
    R = basins.copy()
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if hasA(g.vs[R[i,j] - 1], "label"):
                R[i,j] = g.vs[R[i,j] - 1]["label"] + 1
            else:
                R[i,j] = 0
    return R

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
def showSegmentation(orig, image, b1, b2):
    fig = plt.figure()
    a = fig.add_subplot(1,4,1)
    plt.imshow(color.hsv2rgb(orig))
    plt.title("Original")
    a = fig.add_subplot(1,4,2)
    plt.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
    plt.title("Gradient")
    a = fig.add_subplot(1,4,3)
    plt.imshow(b1, cmap=plt.cm.spectral, interpolation='nearest')
    a.set_title("Watershed")
    a = fig.add_subplot(1,4,4)
    plt.imshow(b2, cmap=plt.cm.spectral, interpolation='nearest')
    a.set_title("Waterfall")
    plt.show()
