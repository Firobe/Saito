import numpy as np
import sys

"""
    Get a feature vector for an image
"""
count = 0
def imageToFeatures(im, nbins=4):
    global count;
    sys.stdout.write("\033[F")
    print ("Computing features : " + str(count))
    count += 4;
    T = np.array([0] * nbins ** 3)
    binWidth = int(256 / nbins)
    n2 = nbins ** 2
    image = np.array(im)
    total = 0
    for row in image[::2]:
        for pixel in row[::2]:
            total += 1
            if(not hasattr(pixel, "__len__")):
                pixel = pixel // binWidth
                T[pixel * (n2 + nbins + 1)] += 1
            else:
                pixel = pixel // binWidth;
                T[pixel[0] * n2 + pixel[1] * nbins + pixel[2]] += 1
    T = T / total
    return T
