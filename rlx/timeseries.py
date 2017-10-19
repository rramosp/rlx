import numpy as np
import rlx.utils as ru
from joblib import Parallel, delayed
import pandas as pd

class TimedateContinuosCoverage_Ranger:
    """
    creates contiguous timedate ranges for train and for test.
    """
    def __init__(self, start_date, end_date,
                 train_period, test_period,
                 min_gap_period="0d", max_gap_period="0d",
                 verbose=False):

        bd_class = pd.tseries.offsets.BusinessDay

        self.start_date = start_date
        self.end_date = end_date
        self.train_period   = ru.to_timedelta(train_period)
        self.test_period    = ru.to_timedelta(test_period)
        self.min_gap_period = ru.to_timedelta(min_gap_period)
        self.max_gap_period = ru.to_timedelta(max_gap_period)
        self.verbose = verbose

    def __iter__(self):
        date = self.start_date
        while date + self.train_period + self.test_period < self.end_date:
            train_dates = (date, date + self.train_period)
            test_dates = (date + self.train_period, date + self.train_period + self.test_period)
            if self.verbose:
                print "TR [", train_dates[0].strftime("%Y-%m-%d %H:%M:%S"), "-", train_dates[1].strftime(
                    "%Y-%m-%d %H:%M:%S"), "] -- TS [", \
                    test_dates[0].strftime("%Y-%m-%d %H:%M:%S"), "-", test_dates[1].strftime("%Y-%m-%d %H:%M:%S"), "]"
            date += self.test_period + self.min_gap_period + (
                                                             self.max_gap_period - self.min_gap_period) * np.random.random()

            yield train_dates, test_dates

def timedate_3way_crossval_runner(estimator, tr_X, val_X, tr_y, val_y, tr_dates, ts_dates, scorer):
    vXtr = tr_X[(tr_X.index >= tr_dates[0]) & (tr_X.index < tr_dates[1])]
    vytr = tr_y[(tr_X.index >= tr_dates[0]) & (tr_X.index < tr_dates[1])]

    vXts = tr_X[(tr_X.index >= ts_dates[0]) & (tr_X.index < ts_dates[1])]
    vyts = tr_y[(tr_X.index >= ts_dates[0]) & (tr_X.index < ts_dates[1])]

    vXval = val_X[(val_X.index >= ts_dates[0]) & (val_X.index < ts_dates[1])]
    vyval = val_y[(val_X.index >= ts_dates[0]) & (val_X.index < ts_dates[1])]

    estimator.fit(vXtr.values, vytr.values)
    tr_score = scorer(estimator, vXtr.values, vytr.values)
    ts_score = scorer(estimator, vXts.values, vyts.values)
    val_score = scorer(estimator, vXval.values, vyval.values)

    return tr_score, ts_score, val_score

def timedate_3way_crossval(estimator, tr_X, val_X, tr_y, val_y, train_period, test_period, scorer,
                           n_jobs=-1, verbose=50):
    """
     three way cross val for time indexed data:
        - train (build model) with elements in period (D1 to D2) of dataset 1
        - test (predict) with elements in period (D2 to D3) of dataset 1
        - val (predict) with elements in period (D2 to D3) of dataset 2

        dataset 1 and 2 should contain data in the same time ranges aprox.
    """

    t = TimedateContinuosCoverage_Ranger(start_date=np.min(tr_X.index), end_date=np.max(tr_X.index),
                                         train_period=train_period, test_period=test_period)
    time_ranges = [i for i in t]
    if verbose>10:
        print "cross validating", len(time_ranges), "time ranges"

    r = ru.mParallel(n_jobs=n_jobs, verbose=verbose)(delayed(timedate_3way_crossval_runner)( \
        estimator,
        tr_X, val_X, tr_y, val_y,
        tr_dates, ts_dates, scorer) \
                                                  for tr_dates, ts_dates in time_ranges)
    return pd.DataFrame(r, columns=["train", "test", "val"], index=pd.Index(range(len(r)), name="iteration"))
