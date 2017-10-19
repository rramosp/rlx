
import itertools
import pandas as pd
from datetime import *
from joblib import Parallel
import sys


def running_in_notebook():
    try:
        cfg = get_ipython().config
        if cfg['IPKernelApp']['parent_appname'] == 'ipython-notebook':
            return True
        else:
            return False
    except NameError:
        return False


class mParallel(Parallel):
    def _print(self, msg, msg_args):
        if self.verbose>10:
            fmsg = '[%s]: %s' % (self, msg % msg_args)
            sys.stdout.write('\r ' + fmsg)
            sys.stdout.flush()

def to_timedelta(t):
    bd_class = pd.tseries.offsets.BusinessDay
    return t if type(t) in [bd_class, pd.Timedelta] else pd.Timedelta(t)


# utc = 1980-01-06UTC + (gps - (leap_count(2014) - leap_count(1980)))
def gpssecs_to_utc(seconds):
    utc = datetime(1980, 1, 6) + timedelta(seconds=int(seconds) - (35 - 19))
    return utc


def gpssecs_to_gpstktime(secs):
    ut = gpssecs_to_utc(secs)
    vt = gpstk.CivilTime()
    vt.day = ut.day
    vt.year = ut.year
    vt.month = ut.month
    vt.hour = ut.hour
    vt.minute = ut.minute
    vt.second = ut.second
    vt.setTimeSystem(gpstk.TimeSystem(gpstk.TimeSystem.GPS))
    return vt.toCommonTime()


def gpstktime_to_gpssecs(t):
    return t.getDays() * 60 * 60 * 24 - 211182767984


def utc_to_gpssecs(t):
    week, sow, day, sod = gpsFromUTC(t.year, t.month, t.day, t.hour, t.minute, t.second, leapSecs=16)
    return week * secsInWeek + sow


def gpssecs_to_gpsday(t):
    dw = gpstk.GPSWeekSecond(gpssecs_to_gpstktime(t))
    return dw.getSOW() / (60 * 60 * 24) + dw.getWeek() * 7


def gpsFromUTC(year, month, day, hour, min, sec, leapSecs=14):
    """converts UTC to: gpsWeek, secsOfWeek, gpsDay, secsOfDay


    from: https://www.lsc-group.phys.uwm.edu/daswg/projects/glue/epydoc/lib64/python2.4/site-packages/glue/gpstime.py

    a good reference is:  http://www.oc.nps.navy.mil/~jclynch/timsys.html

    This is based on the following facts (see reference above):

    GPS time is basically measured in (atomic) seconds since
    January 6, 1980, 00:00:00.0  (the GPS Epoch)

    The GPS week starts on Saturday midnight (Sunday morning), and runs
    for 604800 seconds.

    Currently, GPS time is 13 seconds ahead of UTC (see above reference).
    While GPS SVs transmit this difference and the date when another leap
    second takes effect, the use of leap seconds cannot be predicted.  This
    routine is precise until the next leap second is introduced and has to be
    updated after that.

    SOW = Seconds of Week
    SOD = Seconds of Day

    Note:  Python represents time in integer seconds, fractions are lost!!!
    """

    secFract = sec % 1
    epochTuple = gpsEpoch + (-1, -1, 0)
    t0 = time.mktime(epochTuple)
    t = time.mktime((year, month, day, hour, min, sec, -1, -1, 0))
    # Note: time.mktime strictly works in localtime and to yield UTC, it should be
    #       corrected with time.timezone
    #       However, since we use the difference, this correction is unnecessary.
    # Warning:  trouble if daylight savings flag is set to -1 or 1 !!!
    t = t + leapSecs
    tdiff = t - t0
    gpsSOW = (tdiff % secsInWeek) + secFract
    gpsWeek = int(math.floor(tdiff / secsInWeek))
    gpsDay = int(math.floor(gpsSOW / secsInDay))
    gpsSOD = (gpsSOW % secsInDay)
    return (gpsWeek, gpsSOW, gpsDay, gpsSOD)


def flatten (x):
    return [i for i in itertools.chain.from_iterable(x)]

class DictionaryList:

    def __init__(self):
        self.d = {}
        self.summarizers = {"mean": np.mean, "std": np.std}

    def add_summarizer(self, name, func):
        self.summarizers[name] = func

    def append(self, label, item):
        if label not in self.d.keys():
            self.d[label] = []
        self.d[label].append(item)

    def append_flat(self, label, item_list):
        if label not in self.d.keys():
            self.d[label] = []

        for i in item_list:
            self.d[label].append(i)

    def summary(self):
        r = {}
        for k in self.d.keys():
            r[k] = {}
            for sk in self.summarizers.keys():
                r[k][sk] = self.summarizers[sk](np.array(self.d[k]))
        return r

    def summary_df(self):
        rs = self.summary()
        df = pd.DataFrame([rs[k].values() for k in rs.keys()], columns=rs[rs.keys()[0]].keys())
        df.index = rs.keys()
        return df.sort_index()

    def get_dataframe(self):
        rf = []
        n  = len(self[self.keys()[0]])
        for k in sorted(self.keys()):
            if len(self[k])!=n:
                raise ValueError("all items in dictionary list must contain the same number of elements")
            rf.append(self[k])
        rf = np.array(rf).T
        rf = pd.DataFrame(rf, columns=sorted(self.keys()))
        return rf

    def __getitem__(self, label):
        return np.array(self.d[label])

    def keys(self):
        return self.d.keys()

    def values(self):
        return self.d.values()

    def plot(self, xseries=None, xlabel="", figsize=(15,3), **kwargs):
        if figsize!=None:
            plt.figure(figsize=figsize)
        for i,k in enumerate(self.d.keys()):
            plt.subplot(1,len(self.d.keys()),i+1)
            plt.plot(range(len(self.d[k])) if xseries is None else xseries,
                     self.d[k], **kwargs)
            plt.xlabel(xlabel)
            plt.title(k)