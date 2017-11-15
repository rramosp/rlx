import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import os
import rlx.utils      as ru
import rlx.math       as rm
import rlx.ml         as rml
import rlx.timeseries as rts
import rlx.kalman     as rk

if ru.running_in_notebook():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

def get_cluster_colors(clusters):
    return {c: plt.cm.brg(255*c/(len(clusters)-1)) for c in clusters}

def load_Ik_data(fname, show_dates=True):
    print "loading file"
    d = pd.read_csv(fname)
    print "converting time"
    k = [0]*len(d)
    for c,i in tqdm(enumerate(d.timestamp), total=len(d)):
        k[c] = ru.gpssecs_to_utc(i)
    d.index = pd.to_datetime(k)
    d = d["2017":]
    d.sort_index(inplace=True)

    rs = []
    if show_dates:
        rs.append("approx dates and SDs in free mode")
        pos = d.groupby("station_id")[["X", "Y", "Z"]].agg([np.mean, np.std, len]).dropna()
        sfree = pos[ ((pos["X", "std"]+pos["Y", "std"]+pos["Z","std"])!=0) & (pos["X", "len"]>1000)]

        for i in sfree.index:
            k = d[d.station_id==i].X.rolling(window=20).mean().dropna()
            k = k[k>10]
            rs.append("%5d"%i + " "+str(k.index[0])+ " -- " +str(k.index[-1]))
        d.columns = ["timestamp", "station_id", "gps_X", "gps_Y", "gps_Z"]
    return d, rs


def compute_positions_and_GPS_errors(d, init_date, calcpos_to):
    dx = d.loc[init_date:calcpos_to]
    dd = d.loc[calcpos_to:]

    # compute fixed position with first period of data
    pos = dx.groupby("station_id")[["gps_X", "gps_Y", "gps_Z"]].agg([np.mean, np.std, len]).dropna()

    # identify stations in free mode
    sfree = pos[((pos["gps_X", "std"] + pos["gps_Y", "std"] + pos["gps_Z", "std"]) != 0) & (pos["gps_X", "len"] > 1000)].copy()
    vfree = {i.name: np.r_[[i["gps_X", "mean"], i["gps_Y", "mean"], i["gps_Z", "mean"]]] for _, i in sfree.iterrows()}

    # identify stations in fixed mode
    sfixed = pos.loc[[i for i in pos.index if i not in sfree.index]]
    vfixed = {i.name: np.r_[[i["gps_X", "mean"], i["gps_Y", "mean"], i["gps_Z", "mean"]]] for _, i in sfixed.iterrows()}

    # compute sd groups and centroids
    X = sfree[[("gps_X", "mean"), ("gps_Y", "mean")]].values
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=6, random_state=0)
    km.fit(X)
    sfree["cluster"] = km.predict(X)
    sd_clusters = {i: np.unique(sfree[sfree.cluster == i].index) for i in np.unique(sfree.cluster)}
    clusters_sd = {i: j for i, j in ru.flatten([[[sd, k] for sd in sd_clusters[k]] for k in sd_clusters.keys()])}

    sds = np.unique(sfree.index.values)
    dd = dd[[i in sds for i in dd.station_id.values]]
    print "datapoints to compute fix positions", len(dx), "for", len(np.unique(dx.station_id)), "SDs (free and fixed)"
    print "datapoints for experiments         ", len(dd), "for", len(np.unique(dd.station_id)), "SDs (in free mode)"

    print "computing GPS position errors"
    diffs = np.r_[
        [(np.r_[i.gps_X, i.gps_Y, i.gps_Z] - vfree[i.station_id]) for _, i in tqdm(dd.iterrows(), total=len(dd))]]
    dd = pd.DataFrame(np.hstack((dd.values, diffs)), columns=list(dd.columns) + ["dX", "dY", "dZ"], index=dd.index)
    dd["station_id"] = dd.station_id.values.astype(int)
    vfree = pd.DataFrame(
        [np.r_[[i["gps_X", "mean"], i["gps_Y", "mean"], i["gps_Z", "mean"], i["cluster"]]] for _, i in
         sfree.iterrows()],
        columns=["X", "Y", "Z", "cluster"], index=sfree.index)
    vfree["cluster"] = vfree.cluster.values.astype(int)

    vfixed = pd.DataFrame(
        [np.r_[[i["gps_X", "mean"], i["gps_Y", "mean"], i["gps_Z", "mean"]]] for _, i in sfixed.iterrows()],
        columns=["X", "Y", "Z"], index=sfixed.index)

    return dd, vfree, vfixed

def plot_sdgroups(vfree, vfixed, legend=True, figsize=(12,8)):
    clusters = np.unique(vfree.cluster)
    cluster_colors = get_cluster_colors(clusters)

    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure()
    for c,i in enumerate(clusters):
        k = vfree[vfree.cluster==i][["Y", "X"]].values
        cpos = vfree[vfree.cluster == c][["X", "Y"]].mean(axis=0)
        plt.text(cpos.Y+500000, cpos.X, "GROUP "+str(i))
        plt.scatter(k[:,0], k[:,1], color=cluster_colors[i], label="free SD group "+str(i), s=50)
    plt.scatter(vfixed["Y"], vfixed["X"], color="blue", alpha=.2, label="SD fixed", s=20)

    for _,i in vfree.iterrows():
        plt.text( i["Y"], i["X"], str(i.name),)
    if legend:
        plt.legend();
    plt.xticks([]); plt.yticks([]);
    plt.axis('equal');
    return fig

def timealign_and_resample(dd, experiment_starts, sampling_period, verbose=True,
                           smooth_windows=None, kalman=True):
    import numbers

    sds = np.unique(dd.station_id.values).astype(int)
    dfree = dd[(dd.index > experiment_starts)]
    dfree = {int(i): dfree[dfree.station_id == i].resample(sampling_period).first().replace('nan', np.nan).fillna(method="pad") \
             for i in np.unique(dfree.station_id)}

    if smooth_windows is not None:
        if isinstance(smooth_windows, numbers.Number):
            smooth_windows = [smooth_windows]

        for wsize in smooth_windows:
            print "smoothing error GPS signal with wsize=",wsize
            for k in tqdm(dfree.values()):
                smooth_cols = ["dX", "dY", "dZ"]
                for c in smooth_cols:
                    k["smooth-%d:"%wsize + c] = k[c].rolling(wsize).mean()
                k.dropna(inplace=True)

    if kalman:
        print "kalman on GPS signal"
        for k in tqdm(dfree.values()):
            kalm_cols = ["dX", "dY", "dZ"]
            for c in kalm_cols:
                k["kalm:" + c] = rk.offline_kf(k[c].values,0.4)
            k.dropna(inplace=True)

    if verbose:
        print "nb of data points per SD in free mode"
        for k,v in dfree.iteritems():
            print "%3d"%k, len(v)


    return dfree


def augment_data_to_reference_stations(dfree, vfree, ref_sds, target_sds):
    """
    #this was used to check whether a target sd was also used as reference

    for r,t in itertools.product(ref_sds, target_sds):
        if r==t:
            raise ValueError("SD %d is both in reference and target sets"%r)
    """
    xvfree = {i.name: i[["X", "Y", "Z"]].values for _, i in vfree.iterrows()}

    # add distance and angle from GPS position to actual position in reference stations
    r = {target_sd: pd.DataFrame([ru.flatten([[xvfree[ref_sd][0]-gps_pos[0],
                                               xvfree[ref_sd][0]-gps_pos[1],
                                               np.linalg.norm(gps_pos - xvfree[ref_sd]),
                                            rm.angle_vector(gps_pos[:2][::-1]-xvfree[ref_sd][:2][::-1])] \
                                           for ref_sd in ref_sds]) \
                                  for gps_pos in dfree[target_sd][["gps_X", "gps_Y", "gps_Z"]].values],
                                 index=dfree[target_sd].index,
                                 columns=ru.flatten([["GDISTX_"+str(k), "GDISTY_"+str(k), "L_" + str(k), "A_" + str(k)] for k in ref_sds])
                                 ).join(dfree[target_sd]) \
         for target_sd in tqdm(target_sds)}

    # add observed GPS error at reference positions
    for target_sd in r.keys():
        for ref_sd in ref_sds:
            cols = ["dX", "dY", "dZ"] + [i for i in dfree[ref_sd].columns if "smooth" in i or "kalm" in i]
            r[target_sd] = r[target_sd].join(dfree[ref_sd][cols], rsuffix="_" + str(ref_sd))


    k = r[r.keys()[0]]
    for sd in r.keys()[1:]:
        k = pd.concat((k,r[sd]))
    k.sort_index(inplace=True)
    return k.dropna()

def testv():
    print rm.angle_vector(np.r_[0,1])


def gps_prediction_experiment(estimator,
                            edata, ref_sds, train_sds, val_sds,
                            train_period, test_period,
                            use_ref_smooth = False,
                            use_ref_rawgpserr = True,
                            use_distances_to_ref=True,
                            use_angles_to_ref=True,
                            use_gpspos_target = True,
                            use_ref_kalman = True,
                            use_XYdistances_to_ref=True,
                            show_cols = False,
                            target_col = "dX",
                            verbose = 50,
                            n_jobs=-1):

    cols = [u'timestamp', u'station_id', u'dX', u'dY', u'dZ'] + \
           ([u'gps_X', u'gps_Y', u'gps_Z'] if use_gpspos_target else []) + \
           [i for i in edata.columns if (i != "station_id") and \
            ("gps" not in i) and \
            (use_ref_smooth or "smooth" not in i) and \
            (use_ref_kalman or "kalm" not in i) and \
            (use_ref_rawgpserr or (not i.startswith("dX_") and not i.startswith("dY_") and not i.startswith("dZ_"))) and \
            (use_distances_to_ref or not i.startswith("L_")) and \
            (use_XYdistances_to_ref or not i.startswith("GDIST")) and \
            (use_angles_to_ref or not i.startswith("A_")) and \
            ("_" in i) and (int(i.split("_")[1]) in ref_sds)]

    train_data = edata[[i in train_sds for i in edata.station_id]][cols]
    val_data   = edata[[i in val_sds for i in edata.station_id]][cols]

    source_cols = [i for i in val_data.columns if i not in [target_col, "dX", "dY", "dZ", "station_id", "timestamp"]]

    tr_X, tr_y   = train_data[source_cols], train_data[target_col]
    val_X, val_y = val_data[source_cols], val_data[target_col]

    if show_cols:
        print "source cols", source_cols
        print "target col", target_col

    scorer = rml.abs_error

    r = rts.timedate_3way_crossval(estimator, tr_X, val_X, tr_y, val_y, train_period, test_period, scorer,
                               n_jobs=n_jobs, verbose=verbose)
    return r

experiment_paramset = ["use_ref_smooth", "use_ref_rawgpserr", "use_distances_to_ref","use_angles_to_ref",
                       "use_gpspos_target", "use_ref_kalman","use_XYdistances_to_ref"]

def gps_prediction_lcurve(estimator, edata, ref_sds, train_target_sds, val_target_sds,
                      use_ref_smooth, use_ref_rawgpserr,
                      use_distances_to_ref,use_angles_to_ref,
                      use_gpspos_target, use_ref_kalman, use_XYdistances_to_ref,
                      train_periods, test_period, target_col = "dX",
                      n_jobs=-1, show_cols=False):
    rc = []

    print "processing", len(train_periods), "train periods"

    for i, train_period in enumerate(train_periods):
        r = gps_prediction_experiment(estimator, edata, ref_sds, train_target_sds, val_target_sds,
                                    use_ref_smooth=use_ref_smooth, use_ref_rawgpserr=use_ref_rawgpserr,
                                    use_distances_to_ref=use_distances_to_ref, use_angles_to_ref=use_angles_to_ref,
                                    use_gpspos_target=use_gpspos_target, use_ref_kalman=use_ref_kalman,
                                    use_XYdistances_to_ref=use_XYdistances_to_ref,
                                    target_col=target_col,
                                    train_period=train_period, test_period=test_period, n_jobs=-1,
                                    show_cols=show_cols if i==0 else False)
        rc.append(r)

    return rc

def plot_lcurve(rc, train_periods):
    valm   = [i.mean().val for i in rc]
    trainm = [i.mean().train for i in rc]
    testm  = [i.mean().test for i in rc]

    vals   = [i.std().val for i in rc]
    trains = [i.std().train for i in rc]
    tests  = [i.std().test for i in rc]

    plt.plot(valm, label="val")
    plt.plot(trainm, label="train")
    plt.plot(testm, label="test")
    plt.legend()
    plt.xticks(range(len(train_periods)), train_periods);
    plt.xlabel("train period")
    plt.ylabel("error (in cm)")
    plt.grid()

def get_baseline(edata, col):
    sds = np.unique(edata.station_id).astype(int)
    r   = np.ones((len(sds), len(sds)))*np.nan
    for i1,i2 in tqdm(itertools.product(range(len(sds)), range(len(sds))),total=len(sds)**2):
        s1 = edata[edata.station_id==sds[i1]][["dX", "dY", "dZ"]]
        s2 = edata[edata.station_id==sds[i2]][["dX", "dY", "dZ"]]
        r[i1,i2] =  np.mean(np.abs(s1[col] - s2[col]))
    r[r==0]=np.nan
    return pd.DataFrame(r, columns=sds, index=sds)


def explore_sds_combinations(estimator, edata, feature_set, test_period="2d", train_period="5d",
                             n_ref_sds=2, max_runs=None, n_jobs=-1):
    import os.path

    fname = "data/" + estimator.__class__.__name__ + "_" + str(
        feature_set) + "_" + train_period + "_" + test_period + "_nrefs_%d"%n_ref_sds + ".csv"
    print "results at ", fname
    params = {i: i in feature_set for i in experiment_paramset}
    sds = np.unique(edata.station_id)

    total = len([i for i in itertools.combinations(sds[1:], n_ref_sds)])
    rcs = []

    pbar = tqdm(total=np.min([total * len(sds), np.inf if max_runs is None else max_runs]))
    count = 1
    for sd in sds:

        p = [i for i in sds if i != sd]
        val_sds = [sd]

        for ref_sds in [i for i in itertools.combinations(p, n_ref_sds)]:

            # check if result already produced in file
            results = pd.read_csv(fname) if os.path.isfile(fname) else None
            if results is None or count not in results["count"].values:

                train_sds = [i for i in sds if i != sd and not i in ref_sds]

                r = gps_prediction_experiment(estimator, edata, ref_sds, train_sds, val_sds,
                                                    test_period=test_period, train_period=train_period,
                                                    show_cols=False, verbose=0, n_jobs=n_jobs, **params)

                r = [[count, sd, ref_sds, train_sds, r.mean().val, r.mean().test, r.mean().train]]
                r = pd.DataFrame(r, columns=["count", "val_sd", "ref_sds", "train_sds", "val_score", "test_score",
                                             "train_score"])

                results = r if results is None else pd.concat((results, r))
                results.to_csv(fname, index=False)
            else:
                print "skipping", count
            count += 1
            pbar.update()
            if max_runs is not None and count > max_runs:
                break

        if max_runs is not None and count > max_runs:
            break
