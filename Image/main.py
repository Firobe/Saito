#!/bin/python
import csv, random, pickle, os.path, multiprocessing, itertools
from joblib import Parallel, delayed
from features import imageToFeatures
from models import *
from config import *

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from sklearn import svm, naive_bayes, metrics
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict

#Emotions :
# P  0 Amusement
# N  1 Anger
# P  2 Awe
# P  3 Content
# N  4 Disgust
# P  5 Excitement
# N  6 Fear
# N  7 Sad
posIndices = [0, 1, 2, 3]
negIndices = [4, 5, 6, 7]
emotionsList = ['Amusement', 'Awe', 'Content', 'Excitement', 'Anger',\
        'Disgust', 'Fear', 'Sadness']

"""
    Returns (filenames, emotions) arrays from a csv file
"""
def getData():
    filenames, emotions = [],[]
    with open(LABEL_PATH, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='\'')
        next(reader, None) #Skip header
        for row in reader:
            filenames.append(row[0])
            emotions.append([int(e) for e in row[1:]])
    return (np.array(filenames), np.array(emotions))

"""
    Converts a full vector of emotions to a vector
    with [positiveSum, negativeSum]
"""
def emotionsFtoNP(emotionsF):
    return [sum([emotionsF[i] for i in posIndices]),\
            sum([emotionsF[i] for i in negIndices])];

"""
    Remove images where the main emotion
    is not decisive (using STRONG_THRESHOLD)
"""
def filterWeakEmotions(emotions):
    return [e.max() > e.sum() * STRONG_THRESHOLD for e in emotions]

"""
    Applies argMax to hot vectors
"""
def compressEmotions(emotions):
    return np.array(list((map(np.argmax, emotions))))

"""
    Transform hot vectors of 8 emotions to hot
    vectors of 2 emotions (N, P)
"""
def emotionsFtoNPlabels(emotionsF):
    emotionsNP = [emotionsFtoNP(e) for e in emotionsF]
    return compressEmotions(emotionsNP)

"""
    Unused
"""
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


"""
    Reads images from filenames
"""
def getImages(filenames):
    return [io.imread(IMDIR + f) for f in filenames]

"""
    Displays a confusion matrix
    Inspired from 'http://scikit-learn.org/stable/auto_examples
        /model_selection/plot_confusion_matrix.html'
"""
def evaluate(prediction, labels, classes, display = True):
    print(metrics.classification_report(
        prediction, labels, target_names=classes))
    print("Accuracy score : ", metrics.accuracy_score(labels, prediction))
    plt.figure()
    M = metrics.confusion_matrix(prediction, labels, range(len(classes))).T
    if display:
        plt.imshow(M, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        thresh = M.max() / 2.
        for i, j in itertools.product(range(M.shape[0]), range(M.shape[1])):
            if(M[i, j] > 0):
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
    return acc / len(labels)

"""
    Returns model, features and labels according to
    parameters in config.py
"""
def getModelAndData():
    #Get the data
    print("Fetching data...")
    filenames, emotionsF = getData();
    if MODE == 'NP':
        labels = emotionsFtoNPlabels(emotionsF)
    else :
        strong = filterWeakEmotions(emotionsF)
        labels = compressEmotions(emotionsF[strong])
        filenames = filenames[strong]
        print(filenames.shape[0], " images selected")

    #Use dumped features if existing
    if FORCE_FEATURE_COMPUTING or not os.path.isfile('dumpFeatures.pickle') :
        #Manually compute features
        print("Computing features...")
        images = getImages(filenames)
        num_cores = multiprocessing.cpu_count()
        #features = np.array(Parallel(n_jobs=num_cores)\
        #        (delayed(imageToFeatures)(i) for i in images))
        features = np.array([imageToFeatures(i) for i in images])
        scaler = StandardScaler();
        print(features.shape)
        scaler.fit_transform(features)
        pickle.dump(features, open('dumpFeatures.pickle', "wb"))
    else:
        #Retrieve previously computed features
        print("Retrieving features...")
        features = pickle.load(open('dumpFeatures.pickle', "rb"))

    print("Testing the model")
    #Select the model
    model = getModel(MODEL_NAME, features, labels)
    return (model, features, labels)

if __name__ == "__main__":
    model, features, labels = getModelAndData()
    prediction = cross_val_predict(model, features, labels, cv = N_SPLITS)
    if MODE == 'NP':
        evaluate(prediction, labels, ['Positive', 'Negative'])
    else:
        evaluate(prediction, labels, emotionsList)
