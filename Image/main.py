#!/bin/python
import csv, random, pickle, os.path, multiprocessing, itertools
from joblib import Parallel, delayed
from features import imageToFeatures

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn import naive_bayes
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics
from sklearn.metrics import label_ranking_average_precision_score

#   'NP' : positive or negative, two classes
#   'F'  : full scale of emotions, eight classes
MODE = 'F'
#   If this is high, only the strongest images will be kept. 0 to disable
STRONG_THRESHOLD = 0.0
#   Number of splits in the K-fold
KSPLITS = 10
#   Compute features no matter if a dump already exists
FORCE_FEATURE_COMPUTING = False
#   Path to images
#IMDIR = 'iaps/'
IMDIR = 'images/'
#   Path to ground truth file
#LABEL_PATH = 'mikels/IAPS.csv'
LABEL_PATH = 'groundTruth.csv'

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

""" Returns (filenames, emotions) arrays from a csv file """
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

def filterWeakEmotions(emotions):
    return [e.max() > e.sum() * STRONG_THRESHOLD for e in emotions]

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

def getImages(filenames):
    return [io.imread(IMDIR + f) for f in filenames]

"""
    Returns a trained model optimized for the NP problem
"""
def NPModel(features, labels):
    #old = 64
    clf = RandomForestClassifier(max_depth=4, n_estimators=64,\
            max_features=20, n_jobs=-1, random_state=39, class_weight \
            = 'balanced')
    clf.fit(features, labels)
    return clf

def FModel(features, labels):
    clf = RandomForestClassifier(max_depth=64, n_estimators=32,\
            max_features=20, n_jobs=-1, class_weight='balanced')
    #clf = svm.SVC()
    #clf = naive_bayes.GaussianNB()
    clf.fit(features, labels)
    print("On training data : ",clf.score(features, labels))
    return clf

"""
    Displays a confusion matrix
    Inspired from 'http://scikit-learn.org/stable/auto_examples
        /model_selection/plot_confusion_matrix.html'
"""
def evaluate(prediction, labels, classes, display = True):
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

def generateTrainedModel():
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
        features = np.array(Parallel(n_jobs=num_cores)\
                (delayed(imageToFeatures)(i) for i in images))
        scaler = StandardScaler();
        print(features.shape)
        scaler.fit_transform(features)
        pickle.dump(features, open('dumpFeatures.pickle', "wb"))
    else:
        #Retrieve previously computed features
        print("Retrieving features...")
        features = pickle.load(open('dumpFeatures.pickle', "rb"))

    print("Testing the model")
    #10-fold testing
    prediction, correct = [], []
    kf = KFold(n_splits = KSPLITS)
    for train_index, test_index in kf.split(features):
        #Train the model
        if MODE == 'NP':
            model = NPModel(features[train_index], labels[train_index])
        else:
            model = FModel(features[train_index], labels[train_index])
        prediction += list(model.predict(features[test_index]))

        #Test the model
        correct += list(labels[test_index])
    return (prediction, correct)

if __name__ == "__main__":
    prediction, correct = generateTrainedModel()
    if MODE == 'NP':
        acc = evaluate(prediction, correct, ['Positive', 'Negative'])
    else:
        #print(prediction)
        #print(label_ranking_average_precision_score(correct, prediction))
        acc = evaluate(prediction, correct, emotionsList)
        simpleP = list(map(lambda x: x in negIndices, prediction))
        simpleC = list(map(lambda x: x in negIndices, correct))
        #acc = evaluate(simpleP, simpleC, ['Positive', 'Negative'])
    print("ACCURACY : ", acc)
