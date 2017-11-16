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

def lcurve(estimator, X, y, scorer, cvs, n_jobs=-1, verbose=0):
    """
       estimator: the estimator
       X,y: the data
       scorer: a scorer object, with signature scorer(estimator, X,y)
       cvlist: list of cross validation objects

       return: a DataFrame

       example:

       cvs = [StratifiedShuffleSplit(n_splits=5, test_size=i)
              for i in [.9,.5,.1]]
       rs = lcurve (cvs=cvs, estimator=DecisionTreeClassifier(),
                    X=X[:100], y=y[:100], n_jobs=5)
    """

    k = [{"cv":cv, "train_idxs": itr, "test_idxs":its} for cv in cvs for itr, its in cv.split(X,y)]
    r = ru.mParallel(n_jobs=n_jobs, verbose=verbose)(delayed(fit_score)(estimator, X, y, i, scorer) for i in k)

    r = pd.DataFrame(r, columns=["cv", "estimator", "train_score", "test_score", "fit_time", "score_time"])
    r = r.groupby([str(i) for i in r.cv]).agg([np.mean, np.std, len])
    r.index = [eval(i) for i in r.index]
    return r

def xlcurve(cvlist, tqdm=None, **kwargs):
    """
       cvlist: list of cross validation objects
       kwargs: args to pass to cross_validate

       return: a DataFrame

       example:

       cvs = [StratifiedShuffleSplit(n_splits=5, test_size=i)
              for i in [.9,.5,.1]]
       rs = lcurve (cvlist=cvs, estimator=DecisionTreeClassifier(),
                    X=X[:100], y=y[:100], n_jobs=5)
    """

    from sklearn.model_selection import cross_validate

    rs = []
    tqdm = list if tqdm is None else tqdm

    for cv in tqdm(cvlist):
        r = cross_validate(cv=cv, return_train_score=True, **kwargs)
        r = pd.DataFrame(r).agg([np.mean, np.std]).T
        rs.append(r)
    idxs = pd.MultiIndex( levels = [rs[0].index, ["mean", "std"]],
                          labels = [ru.flatten([[i]*2 for i in range(len(rs[0]))]), [0,1]*2*rs[0].shape[1]],
                          names  = ["metric", "stat"] )
    k = pd.DataFrame(np.zeros((rs[0].shape[0]*rs[0].shape[1], len(rs))),
                columns=pd.Index(range(len(rs)),name="run"), index=idxs)

    for j,i in enumerate(rs):
        for c in i.index:
            k.loc[c,"mean"][j] = i.loc[c]["mean"]
            k.loc[c,"std"][j] = i.loc[c]["std"]
    k.columns = pd.Index(cvlist, name="run")
    return k


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
    cname = cvs[0].__class__.__name__

    attrs = ["test_size", "n_spxlits", "n_folds"]

    attr = np.r_[[hasattr(cvs[0], i) for i in attrs]]

    if len(np.argwhere(attr))==0:
        xlabels = range(len(cvs))
    else:
        attr = attrs[np.argwhere(attr)[0][0]]
        xlabels = [getattr(i, attr) for i in cvs]
    if isinstance(xlabels[0], int):
        xlabels = ["%d"%i for i in xlabels]
    elif isinstance(xlabels[0], float):
        xlabels = ["%.2f"%i for i in xlabels]

    plt.xticks(range(len(xlabels)), xlabels)
    plt.xlabel(attr)
