#   'NP' : positive or negative, two classes
#   'F'  : full scale of emotions, eight classes
MODE = 'F'
#   If this is high, only the strongest images will be kept. 0 to disable
STRONG_THRESHOLD = 0.
#   Compute features no matter if a dump already exists
FORCE_FEATURE_COMPUTING = False
#   If the feature computing should be done on multiple cores
PARALLEL_FEATURES = True
#   Path to images
#IMDIR = 'iaps/'
IMDIR = 'images/'
#   Path to ground truth file
#LABEL_PATH = 'mikels/IAPS.csv'
LABEL_PATH = 'groundTruth.csv'
#   Number of splits in cross-validation
N_SPLITS = 5
#   Name of model function
#   The called function will be '{MODE}_{MODEL_NAME}'
MODEL_NAME = 'OVR'
