from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import (train_test_split, 
                                     cross_val_score, KFold 
                                    ,GridSearchCV, learning_curve, 
                                    RandomizedSearchCV)
from sklearn.svm import (SVC, 
                         LinearSVC)
from sklearn.metrics import (auc, r2_score, balanced_accuracy_score,
                             log_loss, classification_report
                            ,roc_curve, roc_auc_score, 
                            confusion_matrix,mean_squared_error)
from sklearn.preprocessing import (StandardScaler, LabelEncoder, 
                                   Normalizer, label_binarize, 
                                   OneHotEncoder)
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import (BorderlineSMOTE, 
                                    SMOTE, 
                                    ADASYN, 
                                    SVMSMOTE)

from utils import (one_hot,
                    gridsearchCV)


# Data load
white = pd.read_csv("./data/winequality-white.csv", sep=";")
red = pd.read_csv("./data/winequality-red.csv", sep=';')

# rename columsn
white.rename(columns={"fixed acidity":'fixed_acidity', "volatile acidity":"vol_acidity", 
                      'citric acid':"citric_acid", "free sulfur dioxide":"free_sul_dioxide", 
                      "total sulfur dioxide":"total_SD", "residual sugar": "resid_sugar"}, inplace=True)
red.rename(columns={"fixed acidity":'fixed_acidity', "volatile acidity":"vol_acidity", 
                    'citric acid':"citric_acid", "free sulfur dioxide":"free_sul_dioxide", 
                    "total sulfur dioxide":"total_SD", "residual sugar": "resid_sugar"}, inplace=True)
cols = white.columns.tolist()

whiteH = white[white["quality"]>=7]
redH = red[red["quality"]>=7]

whiteM = white[(white["quality"]>=5) & (white["quality"]<=6)]
whiteL = white[white['quality']<=4]

redM = red[(red["quality"]>=5) & (red["quality"]<=6)]
redL = red[red['quality']<=4]

# Group the wine quality [low to mid to high]
whiteH.loc[:,'group'] = 'H'
redH.loc[:,'group'] = 'H'
whiteM.loc[:,'group'] = 'M'
whiteL.loc[:,'group'] = 'L'
redM.loc[:,'group'] = 'M'
redL.loc[:,'group'] = 'L'
redn = redM.append(redL)
redn = redn.append(redH)
whiten = whiteM.append(whiteL)
whiten = whiten.append(whiteH)

# label encoding of groups
# encode labels with value between 0 and n_classes-1.
le = LabelEncoder()

# force pandas to behave like a dataframe by passing a list of columns for the one hot
Yw = whiten.group.values #
# Yw = le.fit_transform(Yw)
Xw = whiten.drop(['group', 'quality'], axis = 1)

# remove features and perform PCA again
__Xw = Xw.copy()
__Xw = __Xw.drop(['resid_sugar', 'chlorides', 'citric_acid', 'alcohol', 'total_SD'], axis = 1)

# Using original labels
# sampling for logistic regression
X_rsW_smt, y_rsW_smt = SMOTE(k_neighbors=4,n_jobs=-1).fit_resample(__Xw, whiten.quality.values)
X_rsW_ada, y_rsW_ada = ADASYN(n_neighbors=4,n_jobs=-1).fit_resample(__Xw, whiten.quality.values)

# Grouping
# sampling for logistic regression
gX_rsW_smt, gy_rsW_smt = SMOTE(k_neighbors=4,n_jobs=-1).fit_resample(__Xw, Yw)
gX_rsW_ada, gy_rsW_ada = ADASYN(n_neighbors=4,n_jobs=-1).fit_resample(__Xw, Yw)
# sampling for support vector machines
gX_rsW_svm, gy_rsW_svm = SVMSMOTE(k_neighbors=4,n_jobs=-1,m_neighbors=4).fit_resample(__Xw, Yw)

# parameters
n_leaf = np.arange(1,6)
depth = np.arange(3,7)
c_space = np.logspace(-1, 1, 4)
n_features_to_test = np.arange(3,7)
k_space = np.arange(1,7)
kernel = ['rbf','poly'] # for linear SVC
degree = [3,4,5]
c_svm = [15,1]
g_space = [1,10]

# start dataframe 
w_results = pd.DataFrame()

# Setup the pipeline
steps = [('scaler', StandardScaler()),
         ('SVM', SVC(decision_function_shape='ovo', kernel='rbf',
                     probability=True, class_weight='balanced', cache_size=800))]
# Specify the hyperparameter space
param_grid = {'SVM__C': c_svm,
              'SVM__gamma' : g_space}
est, cls, scr = gridsearchCV(__Xw, whiten.quality.values,steps,param_grid, plt_mat=True)
df_scratch = pd.DataFrame(data={'Estimator':type(est.steps[1][1]),
                                "Transformation":est.steps[0][0], 'Sampling':'no',
                                'Classes':'Original','Balanced Accuracy':scr}, index=[0])
w_results = w_results.append(df_scratch)

