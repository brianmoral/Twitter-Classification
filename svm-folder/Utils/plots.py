import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import seaborn as sns


## makes meshgrid in plotting_PCA_SVC
def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

## makes contours in plotting_PCA_SVC
def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

################################
# We plot the SVC according to how many 
# PCA components we have 
#
# Params: 
#   X := Features (Components returned from PCA)
#   Y := Lables (Tones)
#   First_c := First component to choose 
#   Second_x := Second component to choose 
#   title  := title for plot
#
# Return: Plots the SVC 
################################
def plotting_PCA_SVC(X, y, first_c, second_c, title = "Pure Data SVC using PCA"):

    first = X[:, first_c].reshape((X.shape[0], 1))
    second = X[:, second_c].reshape((X.shape[0], 1))
    X = np.concatenate((first, second), axis = 1)

    model = svm.SVC(kernel='linear')
    clf = model.fit(X, np.ravel(y))

    fig, ax = plt.subplots()
    # Set-up grid for plotting.
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_ylabel('PC' + str(second_c))
    ax.set_xlabel('PC' + str(first_c))
    ax.set_xticks((-0.085, 0.085))
    ax.set_yticks((-0.085, 0.085))
    ax.set_title(title)
    ax.legend()
    plt.axis([-0.085, 0.085, -0.085, 0.085])
    plt.savefig('Pure-PCA_Brian.png')


def averaging_plot(accuracy_df, x_label, title: str):
    mean = accuracy_df.mean()
    std = accuracy_df.std()

    sns.set_theme()
    plt.figure(figsize=(10,8))
    bp = sns.boxplot(data=accuracy_df, showmeans=True,
                meanprops={
                    "marker": "o",
                    "markerfacecolor": "blue",
                    "markeredgecolor": "black",
                    "markersize": "4"
                })
    for i in range(len(mean)):
        bp.annotate(
            ' μ={:.2f}\n σ={:.2f}'.format(mean[i], std[i]), 
            xy=(i, mean[i]), 
            horizontalalignment='center'
        )
    plt.ylabel("Accuracy", size= 14)
    plt.xlabel(x_label, size = 14)
    plt.title(title, size = 18)