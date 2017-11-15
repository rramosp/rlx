import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import rlx.auger as auger
import deepdish as dd
from tqdm import tqdm_notebook as tqdm
from rlx.ml import *
import rlx.utils as rxu


def get_data(result_list):
    ers = [pd.read_csv(i) for i in result_list]
    ens = [i.split("/")[1] for i in result_list]
    vfree = pd.read_csv("data/vfree.csv")
    vfree.set_index(vfree.station_id, inplace=True)
    ecols = [plt.cm.gist_rainbow(255 * c / (len(ens) - 1)) for c in range(len(ens))]

    return ers, ens, vfree, ecols

def plot_1(ers, ens, vfree, ecols):
    plt.figure(figsize=(20,4))
    for en,er in zip(ens, ers):
        k = er.groupby("val_sd")["val_score"].mean()
        plt.plot(k.values, label=en,alpha=.7)
    plt.xticks(range(len(k)), [int(i) for i in k.index])
    plt.legend();
    plt.xlabel("validation sd")
    plt.ylabel("error in cm")
    plt.grid()


def plot_2(ers, ens, vfree, ecols):
    plt.figure(figsize=(20,4))
    for en,er in zip(ens, ers):
        k = er.groupby("ref_sds")["val_score"].mean()
        k.index = [int(eval(i)[0]) for i in k.index.values]
        k.sort_index(inplace=True)
        plt.plot(k.values, label=en,alpha=.7)
    plt.xticks(range(len(k)), k.index.values)
    plt.legend();
    plt.xlabel("reference sd")
    plt.ylabel("error in cm")
    plt.grid()

def plot_cross_refval_results(l, title="", with_std=True, with_mean=True, vmin=None, vmax=None):
    from rlx import utils as ru
    sds_val = np.unique(l.val_sd).astype(int)
    sds_ref = np.unique(ru.flatten([eval(i) for i in l.ref_sds])).astype(int)

    """
    r = pd.DataFrame([ ['[]']*len(sds_ref)]*len(sds_val),
                     columns=pd.Series(sds_ref, name="ref_sd"),
                     index=pd.Index(sds_val, name="val_sd"))

    for _,i in tqdm(l.iterrows(), total=len(l)):
        for ref_sd in eval(i.ref_sds):
            r.loc[i.val_sd, ref_sd]=str(eval(r.loc[i.val_sd,ref_sd])+[i.val_score])
    """

    import itertools
    t = {str((i, j)): [] for i, j in itertools.product(sds_val, sds_ref)}
#    print t.keys()
    for _, item in tqdm(l.iterrows(), total=len(l)):
        for sd in eval(item.ref_sds):
#            print str((int(item.val_sd), int(sd)))
            t[str((int(item.val_sd), int(sd)))].append(item.val_score)

    rm = pd.DataFrame(np.zeros((len(sds_val), len(sds_ref))), index=pd.Index(sds_val, name="val_sd"),
                      columns=pd.Series(sds_ref, name="ref_sd"))
    rs = rm.copy()
    r  = rm.copy()
    for i, j in itertools.product(sds_val, sds_ref):
        rm.loc[i, j] = np.mean(t[str((i, j))])
        rs.loc[i, j] = np.std(t[str((i, j))])
        r.loc[i, j] = str(t[str((i, j))])

    def f(x, function):
        x = eval(x)
        return np.nan if len(x)==0 else function(x)

    rm = r.apply(lambda x:[f(i, np.mean) for i in x])
    rs = r.apply(lambda x:[f(i, np.std) for i in x])

    if with_std and with_mean:
        plt.figure(figsize=(12,6))
        plt.subplot(121)
    if with_mean:
        plt.imshow(rm, origin="bottom", interpolation="none", vmin=vmin, vmax=vmax, cmap=plt.cm.gnuplot)
        plt.colorbar(fraction=.04)
        plt.xticks(range(len(rm.columns)), [int(i) for i in rm.columns], rotation="vertical");
        plt.yticks(range(len(rm.index)), [int(i) for i in rm.index]);
        plt.xlabel("ref sd")
        plt.ylabel("target sd")
        plt.title("mean err\n"+"\n".join(rxu.split_str(title,45)))

    if with_std and with_mean:
        plt.subplot(122)
    if with_std:
        plt.imshow(rs, origin="bottom", interpolation="none", vmin=vmin, vmax=vmax, cmap=plt.cm.gnuplot)
        plt.colorbar(fraction=.04)
        plt.xticks(range(len(rs.columns)), [int(i) for i in rs.columns], rotation="vertical");
        plt.yticks(range(len(rs.index)), [int(i) for i in rs.index]);
        plt.xlabel("ref sd")
        plt.ylabel("target sd")
        plt.title("std err\n"+"\n".join(rxu.split_str(title,45)))

    return r,rm,rs

def plot_3(ers, ens, vfree, ecols):
    summs = []
    plt.figure(figsize=(len(ens)*7,5))
    for i,(en,er) in enumerate(zip(ens, ers)):
        plt.subplot(1,len(ens),i+1)
        summs.append(plot_cross_refval_results(er, with_std=False, title=en, vmin=35,vmax=80))
    return summs

def plot_4(ers, ens, vfree, ecols):
    summs = []
    plt.figure(figsize=(len(ens)*7,5))
    for i,(en,er) in enumerate(zip(ens, ers)):
        plt.subplot(1,len(ens),i+1)
        summs.append(plot_cross_refval_results(er, with_mean=False, title=en, vmin=0, vmax=2))

def plot_results_per_sd(lrm, lrs,  color, label, axis=1, plot_std=True):
    m,s = lrm.mean(axis=axis).values, lrs.mean(axis=axis).values
    plt.plot(range(len(m)), m, color=color, label=label,alpha=.8)
    if plot_std:
        plt.fill_between(range(len(m)), m-s,m+s, alpha=.2, color=color)
    plt.xlim(-.3,len(m)-.3)
    plt.xticks(range(len(m)), lrm.index if axis==1 else lrm.columns);
    plt.xlabel(lrm.index.name if axis==1 else lrm.columns.name)
    plt.ylabel("prediction error (cm)")


def summarize_clusters_by_cols(lrm, vfree):
    clusters = {k: eval(v) for k,v in dict(vfree.groupby("cluster").station_id.agg(lambda x: str([i for i in x.values]))).iteritems()}
    return pd.concat([pd.DataFrame(lrm[[i for i in clusters[c]]].mean(axis=1), \
                        columns=["cluster_%d"%c]) \
                      for c in clusters.keys()], axis=1)

def summarize_clusters_by_rows(lrm, vfree):
    clusters = {k: eval(v) for k,v in dict(vfree.groupby("cluster").station_id.agg(lambda x: str([i for i in x.values]))).iteritems()}
    r = pd.concat([pd.DataFrame(lrm.loc[[i for i in clusters[c]]].mean(axis=0)) \
                          for c in clusters.keys()], axis=1).T
    r.index = pd.Index(["cluster_%d"%i for i in clusters.keys()], name="cluster")
    return r


def plot_5(ers, ens, vfree, ecols, summs):
    clusters = {k: eval(v) for k, v in
                dict(vfree.groupby("cluster").station_id.agg(lambda x: str([i for i in x.values]))).iteritems()}
    vmin, vmax = 35, 70
    plt.figure(figsize=(len(ens) * 7, 5))
    for i, ((r, rm, rs), en, er) in enumerate(zip(summs, ens, ers)):
        plt.subplot(1, len(ens), i + 1)
        ss = summarize_clusters_by_rows(summarize_clusters_by_cols(rm, vfree), vfree)
        plt.imshow(ss, origin="bottom", interpolation="none", vmin=vmin, vmax=vmax, cmap=plt.cm.gnuplot)
        plt.xticks(range(len(clusters)), clusters.keys(), rotation="vertical")
        plt.yticks(range(len(clusters)), clusters.keys())
        plt.xlabel("ref cluster")
        plt.ylabel("val cluster")
        plt.colorbar(fraction=.05)
        plt.title("\n".join(rxu.split_str(en, 30)))

def get_distances_matrix(vfree):
    sds = vfree.station_id.values
    r = pd.DataFrame(np.zeros((len(sds), len(sds))), columns=sds, index=sds)
    for i,j in itertools.product(vfree.station_id.values, vfree.station_id.values):
        r.loc[i,j] = np.linalg.norm(vfree.loc[i][["X", "Y"]].values - vfree.loc[j][["X", "Y"]].values)
    return r

def plot_distance_vs_error(lrm, vfree, title, val_clusters=None, show_legend=True, groupby="val"):
    assert val_clusters is not None, "must specify set of clusters to show"
    ds = get_distances_matrix(vfree)
    cmap = plt.cm.gist_rainbow
    cmap = plt.cm.gnuplot
    if groupby=="val":
        sds = vfree[[i in val_clusters for i in vfree.cluster ]].index.values
        sds_colors = {sd:cmap(col) for sd,col in zip(sds, np.linspace(0,1,len(sds)))}
        for sd, col in sds_colors.iteritems():
            plt.scatter(0,0,  color=col, alpha=.5, label="target sd %d"%sd)
    else:
        clusters = np.unique(vfree.cluster.values)
        ccols = { cluster: cmap(color) for cluster, color in zip(clusters, np.linspace(0,1,len(clusters)))}
        sds_colors = {sd: ccols[cluster] for sd,cluster in zip(vfree.station_id, vfree.cluster)}
        for cluster, col in ccols.iteritems():
            plt.scatter(0,0,  color=col, alpha=.5, label="ref cluster %d"%cluster)
    clusters = np.unique(vfree.cluster)
    for i,c in itertools.product(lrm.index, lrm.columns):
        if i!=c:
            if vfree.loc[i].cluster in val_clusters:
                color = sds_colors[i] if groupby=="val" else sds_colors[c]
                plt.scatter(lrm.loc[i,c], ds.loc[i,c]/1e5, color=color, edgecolors="black", s=100, alpha=.5)
                if vfree.loc[i].cluster == vfree.loc[c].cluster:
                    plt.scatter(lrm.loc[i,c], ds.loc[i,c]/1e5, edgecolors="black",
                                facecolors="none", alpha=.7, s=200, linewidths=1)
#                    plt.text(lrm.loc[i,c]+.5, ds.loc[i,c]/1e5, str(int(i)))
    plt.scatter(0,0, edgecolors="black",
                facecolors="none", alpha=.2, s=100, linewidths=1, label="ref + val\nin same cluster")
    plt.xlim(np.nanmin(lrm.values)*.97, np.nanmax(lrm.values)*1.03)
    plt.ylim(np.nanmin(ds.values)*.97/1e5, np.nanmax(ds.values)*1.03/1e5)
    plt.ylabel("distance between val and ref (km)")
    plt.xlabel("prediction error (cm)")
    plt.title("\n".join(rxu.split_str(title,30)))
    plt.grid()
    if show_legend:
        plt.legend(loc="center left", bbox_to_anchor=(1.01,0.5), title="clusters "+str(val_clusters))


def plot_6(ers, ens, vfree, ecols, summs):
    cpos = vfree.groupby("cluster")[["X", "Y", "Z"]].mean()
    ds = get_distances_matrix(vfree)
    for cset in [[i] for i in range(6)]:
        plt.figure(figsize=(len(ens) * 6, 5))
        for i, ((r, rm, rs), en, er) in enumerate(zip(summs, ens, ers)):
            plt.subplot(1, len(ens), i + 1)
            plot_distance_vs_error(rm, vfree, en, val_clusters=cset, show_legend=i == (len(ens) - 1), groupby="val")
