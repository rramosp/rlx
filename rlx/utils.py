
import itertools
import pandas as pd
import numpy as np
from datetime import *
from joblib import Parallel
import sys
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
import re
import inspect
from IPython.core import display
import contextlib
import math

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


def split_str(s, w):
    """
    splits a string in fixed size splits (except, possibly, the last one)
    :param s: string to split
    :param w: length of splits
    :return: the string splitted
    """
    return [s[w * i:np.min((len(s), w * (i + 1)))] for i in range(len(s) / w + 1 * (len(s) % w != 0))]


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

def show_source(f):
    import inspect
    src = inspect.getsource(f)
    html = hilite_code(src)
    return display.HTML(html)

def hilite_code(code):
    lexer = 'python'
    style = 'colorful'
    defstyles = 'overflow:auto;width:auto;'
    divstyles = ""
    prestyles = """
    margin: 0;
    background: #555;
    background-image: -webkit-linear-gradient(#FFFFFF 50%, #F9F9F9 50%);
    background-image:    -moz-linear-gradient(#FFFFFF 50%, #F9F9F9 50%);
    background-image:     -ms-linear-gradient(#FFFFFF 50%, #F9F9F9 50%);
    background-image:      -o-linear-gradient(#FFFFFF 50%, #F9F9F9 50%);
    background-image:         linear-gradient(#FFFFFF 50%, #F9F9F9 50%);
    background-position: 0 0;
    background-repeat: repeat;
    background-size: 4.5em 2.5em;
        """
    formatter = HtmlFormatter(style=style,
                              linenos=False,
                              noclasses=True,
                              cssclass='',
                              cssstyles=defstyles + divstyles,
                              prestyles=prestyles)
    html = highlight(code, get_lexer_by_name(lexer), formatter)
    nbs = "\n".join(["%4d"%i for i in range(1,len(code.split("\n")))])
    html = "<table><tr><td valign='top'><pre style='margin: 0; line-height: 125%;"+prestyles+"'>"+nbs+"</pre></td><td><div style='text-align: left'>"+html+"</div></td></tr></table>"
    html = "<!-- HTML generated using hilite.me -->" + html
    return html

def get_default_style():
    return 'border:solid gray;border-width:.1em .1em .1em .8em;padding:.2em .6em;'


@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


def np2str(x, dec_places=2, margin=0, linewidth=140):
    int_places = int(np.log10(np.max(np.abs(x))))+1
    fmt = '{: '+str(int_places+dec_places+2)+"."+str(dec_places)+'f}'
    with printoptions(formatter={'float': fmt.format}, linewidth=linewidth):
        s = str(x)
    m = " "*margin
    return m+s.replace("\n", "\n"+m)


def mkdir(newdir):
    import os
    """works the way a good mkdir should :)
        - already exists, silently complete
        - regular file in the way, raise an exception
        - parent directory(ies) does not exist, make them as well
    """
    if os.path.isdir(newdir):
        pass
    elif os.path.isfile(newdir):
        raise OSError("a file with the same name as the desired "\
                      "dir, '%s', already exists." % newdir)
    else:
        head, tail = os.path.split(newdir)
        if head and not os.path.isdir(head):
            mkdir(head)
        if tail:
            os.mkdir(newdir)
