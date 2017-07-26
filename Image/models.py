from sklearn import svm, naive_bayes, metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
from config import *

################################
#   Returns non-trained models
#   Functions should be named 
#   M_N
#   With M begin the mode
#       (either 'F' or 'NP')
#   And N being the name
#       (cf. MODEL_NAME)
################################


def NP_RandomForest(features, labels):
    #old = 64
    clf = RandomForestClassifier(max_depth=4, n_estimators=64,
            max_features=20, n_jobs=4, random_state=39, class_weight
            = 'balanced')
    return clf

def F_RandomForestGrid(features, labels):
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

def F_OVR(features, labels):
    return OneVsRestClassifier(GaussianNB(), n_jobs = 4)

def getModel(name, features, labels):
    return globals()[name](features, labels)
