import numpy as np
import pandas as pd
import rlx.utils as ru
import matplotlib.pyplot as plt
from joblib import delayed

def abs_error(estimator, X, y):
    preds = estimator.predict(X)
    return np.mean(np.abs(y-preds))


class BaselinePredictor:
    """
    chooses the source column with the values closes to the prediction
    """

    def fit(self, X, y):
        self.col = np.argmin([np.mean(np.abs(X[:, i] - y)) for i in range(X.shape[1])])

    def predict(self, X):
        return X[:, self.col]

    def score(self, X, y):
        return abs_error(self, X,y)

def fit_score(estimator, X,y, fold_spec, scorer):
    from sklearn.base import clone
    from time import time
    e = clone(estimator)
    itr, its = fold_spec["train_idxs"], fold_spec["test_idxs"]

    Xtr, ytr = X[itr], y[itr]
    Xts, yts = X[its], y[its]

    t1 = time()
    e.fit(Xtr, ytr)
    fit_time = time()-t1

    t2 = time()
    tr_score = scorer(e, Xtr, ytr)
    ts_score = scorer(e, Xts, yts)
    score_time = time()-t2

    return fold_spec["cv"], estimator, tr_score, ts_score, fit_time, score_time


def lcurve(estimator, X, y, scorer, cvs, n_jobs=-1, verbose=0):
    """
    Parameters:
    -----------
    estimator : the estimator

    X,y : the data

    scorer : a scorer object, with signature scorer(estimator, X,y)

    cvlist : list of cross validation objects

    Returns:
    --------
    return : a DataFrame

    Examples:
    ---------

       cvs = [StratifiedShuffleSplit(n_splits=5, test_size=i)
              for i in [.9,.5,.1]]
       rs = lcurve (cvs=cvs, estimator=DecisionTreeClassifier(),
                    X=X[:100], y=y[:100], n_jobs=5)
    """

    k = [{"cv":cv, "train_idxs": itr, "test_idxs":its} for cv in cvs for itr, its in cv.split(X,y)]
    r = ru.mParallel(n_jobs=n_jobs, verbose=verbose)(delayed(fit_score)(estimator, X, y, i, scorer) for i in k)

    r = pd.DataFrame(r, columns=["cv", "estimator", "train_score", "test_score", "fit_time", "score_time"])
    return r
    keys = {str(i): i for i in r.cv}
    r = r.groupby(keys.keys()).agg([np.mean, np.std, len])
    r.index = [keys[i] for i in r.index]
    return r


def plot_lcurve(rs):
    tsm, tss = rs["test_score"]["mean"].values, rs["test_score"]["std"].values
    trm, trs = rs["train_score"]["mean"].values, rs["train_score"]["std"].values
    plt.plot(tsm, color="red", label="test", marker="o")
    plt.fill_between(range(len(tsm)), tsm-tss, tsm+tss, color="red", alpha=.1)
    plt.plot(trm, color="green", label="train", marker="o")
    plt.fill_between(range(len(trm)), trm-trs, trm+trs, color="green", alpha=.1)
    plt.ylim(0,1.05)
    plt.legend()
    plt.grid()
    plt.ylabel("score")

    cvs = rs.index

    attrs = ["test_size", "n_spxlits", "n_folds"]

    attr = np.r_[[hasattr(cvs[0], i) for i in attrs]]

    if len(np.argwhere(attr)) == 0:
        xlabels = range(len(cvs))
    else:
        attr = attrs[np.argwhere(attr)[0][0]]
        xlabels = [getattr(i, attr) for i in cvs]
    if isinstance(xlabels[0], int):
        xlabels = ["%d" % i for i in xlabels]
    elif isinstance(xlabels[0], float):
        xlabels = ["%.2f" % i for i in xlabels]

    plt.xticks(range(len(xlabels)), xlabels)
    plt.xlabel(attr)
