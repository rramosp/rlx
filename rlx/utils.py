
import itertools
import pandas as pd
import numpy as np
from datetime import *
from joblib import Parallel
import sys
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.core import display
import contextlib
import math
import progressbar
import time
import os
import psutil
import gc
from pandas.api.types import is_string_dtype

STANDARD_COLORS = np.r_[[
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]]


def running_in_notebook():
    try:
        cfg = get_ipython().config
        if cfg['IPKernelApp']['parent_appname'] == 'ipython-notebook':
            return True
        else:
            return False
    except NameError:
        return False


def read_password_protected_zip(fname):
    import getpass, subprocess
    pwd = getpass.getpass("input password for %s: "%fname)
    cmd = "unzip -c -P%s %s"%(pwd, fname)
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
    out, err = p.communicate()
    out = "\n".join(out.split("\n")[2:]+["\n"]*2).strip()
    if len(err)>0:
        raise ValueError("error unzipping encrypted file: "+err)
    return out

def aws_S3_login(aws_credentials_zip_file):
    import json, boto3
    s = read_password_protected_zip(aws_credentials_zip_file)
    aws = json.loads(s)

    s3 = boto3.client(
        's3',
        aws_access_key_id=aws["aws_access_key"],
        aws_secret_access_key=aws["aws_secret_key"]
    )
    return s3

def fix_unicode_columns(d):
    for col in d.columns:
        if is_string_dtype(d[col]):
            d[col] = [i.decode('utf8', 'ignore') for i in d[col]]
    return d

class mParallel(Parallel):
    def _print(self, msg, msg_args):
        if self.verbose > 10:
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

def command(cmd):
    import os
    z = os.system(cmd + " > /tmp/k.stdout 2> /tmp/k.stderr")
    with open("/tmp/k.stdout", "rb") as f:
        stdout = [i.rstrip() for i in f.readlines()]
    with open("/tmp/k.stderr", "rb") as f:
        stderr = [i.rstrip() for i in f.readlines()]
    os.system("rm /tmp/k.stdout /tmp/k.stderr")

    return z, stdout, stderr


def getmem(keys=["rss", "vms"], as_text=True, do_gc=False):
    """
    gets the memory currently used by the running Python process
    see process.memory_info().__dict__ for additional keys

    do_gc: perform garbage collection before measuring memory
    """
    if do_gc:
        gc.collect()
    process = psutil.Process(os.getpid())
    k = process.memory_info()
    k = {i: humanbytes(k.__dict__[i]) if as_text else k.__dict__[i] for i in keys}
    if len(keys) == 1:
        return k[keys[0]]
    else:
        return k


def humanbytes(B):
    'Return the given bytes as a human friendly KB, MB, GB, or TB string'
    B = float(B)
    KB = float(1024)
    MB = float(KB ** 2)  # 1,048,576
    GB = float(KB ** 3)  # 1,073,741,824
    TB = float(KB ** 4)  # 1,099,511,627,776

    if B < KB:
        return '{0} {1}'.format(B, 'Bytes' if 0 == B > 1 else 'Byte')
    elif KB <= B < MB:
        return '{0:.2f} KB'.format(B/KB)
    elif MB <= B < GB:
        return '{0:.2f} MB'.format(B/MB)
    elif GB <= B < TB:
        return '{0:.2f} GB'.format(B/GB)
    elif TB <= B:
        return '{0:.2f} TB'.format(B/TB)


def pbar(**kwargs):
    sys.stdout.flush()
    sys.stderr.flush()
    time.sleep(.2)
    return progressbar.ProgressBar(**kwargs)


class _s:
    """
    substitute in sympy expression expr the symbols in matrix sm with values
    in numpy matrix nm. For instance:

    W      = sy.Matrix(sy.MatrixSymbol("W", 2, 2))
    W_vals = np.random.random(W.shape)
    _s(W**2)._s(W, W_vals).expr

    """
    def __init__(self, expr):
        self.expr = expr

    def _s(self, sm,nm):
        self.expr = self.expr.subs({i[1]:nm.reshape(sm.shape)[i[0]] for i in np.ndenumerate(sm)})
        return self


def plot_heatmap(x, y, x_range=None, y_range=None, grid_size=(5,5),
                 n_levels=20, margin=.01, **kwargs):
    if x_range is None:
        mx = np.abs(np.max(x)-np.min(x))*margin
        x_range = np.min(x)-mx, np.max(x)+mx

    x_bins = np.linspace(x_range[0], x_range[1], grid_size[0]+1)

    if y_range is None:
        my = np.abs(np.max(y)-np.min(x))*margin
        y_range = np.min(y)-my, np.max(y)+my

    y_bins = np.linspace(y_range[0], y_range[1], grid_size[1]+1)

    x_discrete = np.digitize(x, x_bins)
    y_discrete = np.digitize(y, y_bins)

    z = np.zeros((len(y_bins), len(x_bins)))
    for xi in range(len(x_bins)):
        for yi in range(len(y_bins)):
            z[yi, xi] = np.sum((x_discrete==xi)*(y_discrete==yi))
    z = z[1:, 1:]
    x_ticks = np.linspace(x_range[0], x_range[1], grid_size[0])
    y_ticks = np.linspace(y_range[0], y_range[1], grid_size[1])
    plt.contourf(x_ticks, y_ticks, z, levels=np.linspace(np.min(z), np.max(z), n_levels+1), **kwargs )


def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def get_http_image(url):
    import requests
    from cStringIO import StringIO
    from PIL import Image

    rq = requests.get(url)
    return np.array(Image.open(StringIO(rq.content)))


def most_common(lst):
    return max(set(lst), key=lst.count)

def rolling_window(img, shape):
    """
    extracts 1-stride rolling windows of shape over the input img (2D numpy array)
    """
    a = img
    s = (a.shape[0] - shape[0] + 1,) + (a.shape[1] - shape[1] + 1,) + shape
    strides = a.strides + a.strides
    return np.lib.stride_tricks.as_strided(a, shape=s, strides=strides)

def most_common_neighbour(img, window_size):
    """
    returns an image where each pixel is replaced with the most frequently
    occurring value within its neighbourhood
    """
    s1,s2 = window_size
    # pad image to ensure result is the same size as input
    z = np.pad(img, ((s1/2,s1-s1/2-1),(s2/2,s2-s2/2-1)), "reflect")
    rw = rolling_window(z,(s1,s2))
    r = np.zeros(img.shape).astype(img.dtype)
    for y,x in itertools.product(range(rw.shape[0]), range(rw.shape[1])):
        r[y,x]=most_common(list(rw[y,x].flatten()))
    return r
