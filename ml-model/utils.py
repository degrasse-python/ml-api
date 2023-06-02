import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import (StandardScaler, 
                                   LabelEncoder)
from sklearn.model_selection import (train_test_split, 
                                     GridSearchCV)
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, 
                             confusion_matrix)



def plot_confusion_matrix(cm, classes, ax,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Arguments
    ---------
    # cm - 
    # classes - 
    # ax - 
    # normalize - (True, False)
    # title - 
    # cmap - plt.cm.Blues

    Returns
    -------
    graph (returns no objects)
    """


    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    #tick_marks = np.arange(len(classes))
    #plt.xticks(tick_marks, classes, rotation=45)
    plt.sca(ax)
    #plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')


def one_hot(mat, num_classes=3):
    """
    Arguments
    ---------
    # mat - matrix 
        shape= , including vector integer which have shape=[batch,] # label present this really an (m x 1) vec  
    # num_classes - num of classes
        shape=[batch_size, ] vector containing scalar true label for row 

    final_size: matrix
        containing true vectors in place of the single scalar

    Returns
    -------
    list:
    A list which is contains the operations to be preformed in model run.
    """
    
    squeezed_vector = np.squeeze(np.int8(mat))
    one_hot = np.zeros((squeezed_vector.size, num_classes))
    one_hot[np.arange(squeezed_vector.size), squeezed_vector] = 1

    return one_hot


def gridsearchCV(x,y, steps, param_grid, 
                    scoring='balanced_accuracy', 
                    split=0.3, printFeatureImportance=False, 
                    cv_folds=5, plt_mat=False):
    """
    Arguments
    ---------
    # x 
    # y  
    # steps
    # param_grid 
    # scoring='balanced_accuracy' 
    # split=0.3, printFeatureImportance=False 
    # cv_folds=5 
    # plt_mat=False

    Returns
    -------
    # Estimator - b_est
    # List - list_classes 
    # Score - b_score
    """
    le = LabelEncoder()    
    y = le.fit_transform(y)
    X_trainr, X_testr, y_trainr, y_testr = train_test_split(x, y, test_size = 0.3, random_state=2)
    pipe = Pipeline(steps)
    uniq = np.unique(y_testr).tolist()
    uniq = [str(x) for x in uniq]
    #Perform cross-validation:
    # Instantiate the GridSearchCV object: logreg_cv
    clfr_cv = GridSearchCV(pipe, param_grid, scoring=scoring, cv=cv_folds)

    # Fit it to the data
    clfr_cv.fit(X_trainr, y_trainr)
    y_p = clfr_cv.predict(X_testr)

    # Print the tuned parameters and score
    print("Model: {}".format(type(clfr_cv.estimator.steps[1][1])))
    print("Tuned Parameters: {}".format(clfr_cv.best_params_)) 
    print("Tuned balanced accuracy {}".format(clfr_cv.best_score_))
    print(classification_report(y_testr, y_p, target_names=uniq))

    if plt_mat:
        fig, ax = plt.subplots()
        cm_ = confusion_matrix(y_testr, y_p)
        plot_confusion_matrix(cm_, classes=uniq, ax=ax,
                              title='Class Matrix')
    b_est = clfr_cv.best_estimator_
    est = clfr_cv.estimator.steps
    list_classes = clfr_cv.classes_
    b_score = clfr_cv.best_score_
    return b_est, list_classes, b_score  


