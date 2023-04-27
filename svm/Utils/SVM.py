from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import numpy as np

##################################
# PCA 
# 
#
##################################
def dimension_reduction_by_PCA(X, components):
    pca = PCA(n_components=components).fit(X)
    # print(pca.explained_variance_)
    return pca.fit_transform(X)

####################################
# Fits an SVC model and runs a prediction 
# test 
#
# Params:
#   X := features (jaccard distance matrix)
#   Y := Labels (Tones)
#   test_size := Training and testing split. Set to %30
#
# Return: Accuracy score of SVC
####################################
def SVC_model(X, y, test_size = 0.3):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)


    model_clf = svm.SVC(kernel='linear').fit(X_train, np.ravel(y_train))
    predicted = model_clf.predict(X_test)

    return accuracy_score(y_test, predicted)