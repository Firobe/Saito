#!/bin/python
import csv, random, pickle, os.path, multiprocessing, itertools
from joblib import Parallel, delayed

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import exposure
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics
from sklearn.metrics import label_ranking_average_precision_score

from fast import *

#Emotions :
# P  0 Amusement
# N  1 Anger
# P  2 Awe
# P  3 Content
# N  4 Disgust
# P  5 Excitement
# N  6 Fear
# N  7 Sad
posIndices = [0, 2, 3, 5]
negIndices = [1, 4, 6, 7]
emotionsList = ['Amusement', 'Anger', 'Awe', 'Content', 'Disgust',\
        'Excitement', 'Fear', 'Sadness']

""" Returns (filenames, emotions) arrays from a csv file """
def getData(truthFile):
    filenames, emotions = [],[]
    with open(truthFile, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='\'')
        next(reader, None) #Skip header
        for row in reader:
            filenames.append(row[0])
            emotions.append([int(e) for e in row[1:]])
    return (filenames, emotions)

"""
    Converts a full vector of emotions to a vector
    with [positiveSum, negativeSum]
"""
def emotionsFtoNP(emotionsF):
    return [sum([emotionsF[i] for i in posIndices]),\
            sum([emotionsF[i] for i in negIndices])]

def compressEmotions(emotions):
    return np.array(list((map(np.argmax, emotions))))

def emotionsFtoNPlabels(emotionsF):
    emotionsNP = [emotionsFtoNP(e) for e in emotionsF]
    return compressEmotions(emotionsNP)

def emotionsFtoH(emotionsF, threshold = 0.6):
    ret = []
    s = sum(emotionsF)
    ts = 0
    sInd = sorted(range(len(emotionsF)), reverse = True,\
            key=lambda k: emotionsF[k])
    for i in range(len(sInd)):
        if ts / s < threshold:
            ret.append(sInd[i])
            ts += emotionsF[sInd[i]]
        else: break
    return ret


def emotionsFtoHlabels(emotionsF):
    classes = list(map(emotionsFtoH, emotionsF))
    print(classes)
    return MultiLabelBinarizer().fit_transform(classes)

def getImages(filenames, dir = 'images/'):
    return [io.imread(dir + f) for f in filenames]

def dumpLabels(filenames, labels):
    for i in range(len(filenames)):
        print(filenames[i], "\t", labels[i])

"""
    Returns a trained model optimized for the NP problem
"""
def NPModel(features, labels):
    clf = RandomForestClassifier(max_depth=4, n_estimators=64,\
            max_features=64, n_jobs=-1, random_state=39, class_weight \
            = 'balanced')
    clf.fit(features, labels)
    return clf

def FModel(features, labels):
    clf = RandomForestClassifier(max_depth=64, n_estimators=32,\
            max_features=32, n_jobs=-1)
    clf.fit(features, labels)
    return clf

"""
    Displays a confusion matrix
    Inspired from 'http://scikit-learn.org/stable/auto_examples
        /model_selection/plot_confusion_matrix.html'
"""
def confusionMatrix(prediction, labels, classes):
    plt.figure()
    M = metrics.confusion_matrix(prediction, labels, range(len(classes)))
    plt.imshow(M, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = M.max() / 2.
    for i, j in itertools.product(range(M.shape[0]), range(M.shape[1])):
        plt.text(j, i, M[i, j],
                 horizontalalignment="center",
                 color="white" if M[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    acc = 0
    for i in range(len(classes)):
        acc += M[i][i]
    print("ACCURACY : ", acc / len(labels))

#MODE
#   'NP' : positive or negative, two classes
#   'F'  : full scale of emotions, eight classes
MODE = 'F'

if __name__ == "__main__":
    #Get the data
    print("Fetching data...")
    filenames, emotionsF = getData('groundTruth.csv');
    if MODE == 'NP':
        labels = emotionsFtoNPlabels(emotionsF)
    else :
        labels = compressEmotions(emotionsF)
    #dumpLabels(filenames, labels)
    #exit()

    #Use dumped features if existing
    if not os.path.isfile('features.pickle') :
        #Manually compute features
        print("Computing features...")
        images = getImages(filenames)
        num_cores = multiprocessing.cpu_count()
        features = np.array(Parallel(n_jobs=num_cores)\
                (delayed(imageToFeatures)(i, 8) for i in images))
        scaler = StandardScaler();
        scaler.fit_transform(features)
        pickle.dump(features, open('features.pickle', "wb"))
    else:
        #Retrieve previously computed features
        print("Retrieving features...")
        features = pickle.load(open('features.pickle', "rb"))

    print("Testing the model")
    #10-fold testing
    prediction, correct = [], []
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(features):
        #Train the model
        if MODE == 'NP':
            model = NPModel(features[train_index], labels[train_index])
        else:
            model = FModel(features[train_index], labels[train_index])
        prediction += list(model.predict(features[test_index]))

        #Test the model
        correct += list(labels[test_index])
    if MODE == 'NP':
        confusionMatrix(prediction, correct, ['Positive', 'Negative'])
    else:
        #print(prediction)
        #print(label_ranking_average_precision_score(correct, prediction))
        confusionMatrix(prediction, correct, emotionsList)
        simpleP = list(map(lambda x: x in negIndices, prediction))
        simpleC = list(map(lambda x: x in negIndices, correct))
        confusionMatrix(simpleP, simpleC, ['Positive', 'Negative'])
