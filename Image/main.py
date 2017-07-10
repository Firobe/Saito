#!/bin/python
import csv, random, pickle, os.path, multiprocessing, itertools
from joblib import Parallel, delayed
from features import imageToFeatures

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from sklearn import svm, naive_bayes, metrics
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict

#   'NP' : positive or negative, two classes
#   'F'  : full scale of emotions, eight classes
MODE = 'F'
#   If this is high, only the strongest images will be kept. 0 to disable
STRONG_THRESHOLD = 0.0
#   Compute features no matter if a dump already exists
FORCE_FEATURE_COMPUTING = False
#   Path to images
#IMDIR = 'iaps/'
IMDIR = 'images/'
#   Path to ground truth file
#LABEL_PATH = 'mikels/IAPS.csv'
LABEL_PATH = 'groundTruth.csv'
#   Number of splits in cross-validation
N_SPLITS = 5

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
    print("On training data : ",clf.score(features, labels))
    return clf

def FModel(features, labels):
    clf = RandomForestClassifier()
    #clf = svm.SVC()
    #clf = naive_bayes.GaussianNB()
    print(features.shape)
    param_grid = {"n_estimators": [16, 32, 64],
              "max_depth": [4, None],
              "max_features": [6, 8, 10, 20],
              "min_samples_split": [2, 3, 4, 5, 10, 20],
              "min_samples_leaf": [5, 10, 15, 20],
              "bootstrap": [True, False],
              "criterion": ["entropy"],
              "class_weight": ["balanced", None]
              }
    search = GridSearchCV(clf, param_grid=param_grid, verbose=2, n_jobs=4,
            cv=KFold(n_splits=N_SPLITS), refit = False)
    search.fit(features, labels)
    print(search.best_params_)
    return RandomForestClassifier(**search.best_params_)

"""
    Displays a confusion matrix
    Inspired from 'http://scikit-learn.org/stable/auto_examples
        /model_selection/plot_confusion_matrix.html'
"""
def evaluate(prediction, labels, classes, display = True):
    print(metrics.classification_report(
        prediction, labels, target_names=classes))
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
    #Select the model
    model = FModel(features, labels)
    return (model, features, labels)

if __name__ == "__main__":
    model, features, labels = getModelAndData()
    prediction = cross_val_predict(model, features, labels, cv = N_SPLITS)
    if MODE == 'NP':
        evaluate(prediction, labels, ['Positive', 'Negative'])
    else:
        evaluate(prediction, labels, emotionsList)
    #    #simpleP = list(map(lambda x: x in negIndices, prediction))
    #    #simpleC = list(map(lambda x: x in negIndices, correct))
    #    #acc = evaluate(simpleP, simpleC, ['Positive', 'Negative'])
