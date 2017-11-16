import numpy as np
import pandas as pd
import rlx.utils as ru

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


def lcurve(cvlist, tqdm=None, **kwargs):
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
