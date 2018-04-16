import numpy as np
import pandas as pd
from rlx import utils
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from joblib import delayed
from sklearn.neighbors import KernelDensity
from skimage.io import imread
from statsmodels.graphics.tsaplots import plot_acf
import itertools
import sympy as sy
import tensorflow as tf
import tflearn
import shapely as sh
import geopandas as gpd
import descartes
from PIL import Image
from rlx import geo
from skimage.morphology import disk
from skimage.filters.rank import enhance_contrast_percentile


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
        return abs_error(self, X, y)


def fit_score(estimator, X, y, fold_spec, scorer):
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
    r = utils.mParallel(n_jobs=n_jobs, verbose=verbose)(delayed(fit_score)(estimator, X, y, i, scorer) for i in k)

    r = pd.DataFrame(r, columns=["cv", "estimator", "train_score",
                                 "test_score", "fit_time", "score_time"])

    k = {str(i): i for i in r.cv}
    r = r.groupby([str(i) for i in r.cv]).agg([np.mean, np.std, len])
    r.index = [k[i] for i in r.index]

    return r


def plot_lcurve(rs):

    # extract size indicator from cross validator
    cvs = rs.index
    attrs = ["train_size", "test_size", "n_splits", "n_folds"]
    attr = np.r_[[hasattr(cvs[0], i) for i in attrs]]

    if len(np.argwhere(attr)) == 0:
        nlabels = np.arange(len(cvs))
    else:
        attr = attrs[np.argwhere(attr)[0][0]]
        nlabels = np.r_[[getattr(i, attr) for i in cvs]]

    if isinstance(nlabels[0], int):
        xlabels = np.r_[["%d" % i for i in nlabels]]
    elif isinstance(nlabels[0], float):
        xlabels = np.r_[["%.2f" % i for i in nlabels]]
    else:
        xlabels = np.r_[["%d" % i for i in range(len(nlabels))]]
        nlabels = np.arange(len(nlabels))

    idxs = np.argsort(-nlabels) if attr not in ["n_splits", "train_size"] else np.argsort(nlabels)
    tsm, tss = rs["test_score"]["mean"].values[idxs], rs["test_score"]["std"].values[idxs]
    trm, trs = rs["train_score"]["mean"].values[idxs], rs["train_score"]["std"].values[idxs]
    plt.plot(tsm, color="red", label="test", marker="o")
    plt.fill_between(range(len(tsm)), tsm-tss, tsm+tss, color="red", alpha=.1)
    plt.plot(trm, color="green", label="train", marker="o")
    plt.fill_between(range(len(trm)), trm-trs, trm+trs, color="green", alpha=.1)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid()
    plt.ylabel("score")

    plt.xticks(range(len(xlabels)), xlabels[idxs])
    plt.xlabel(attr)


def kldiv_distribs(stats_distribution1, stats_distribution2, x_range):
    a = stats_distribution1.pdf(x_range)
    b = stats_distribution2.pdf(x_range)
    return kldiv(a, b)


def kldiv(a, b):
    idxs = (b > 0) & (b < np.inf)
    return np.sum(a[idxs] * np.log(a[idxs] / b[idxs]))


def mcmc(n_samples, s, q_sampler, q_pdf, init_state_sampler,  use_logprobs=False, verbose=True):
    xi = init_state_sampler()
    r = [xi]
    c = 0
    loop_values = range(n_samples)
    for i in utils.pbar()(loop_values) if verbose else loop_values:
        proposed_state = q_sampler(xi)
        if use_logprobs:
             acceptance_probability = np.exp(s(proposed_state)-s(xi) + q_pdf(proposed_state,xi) - q_pdf(xi,proposed_state))
        else:
             acceptance_probability = s(proposed_state)/s(xi) * q_pdf(proposed_state,xi) / q_pdf(xi,proposed_state)
        acceptance_probability = np.min((1, acceptance_probability))
        if np.random.random()<acceptance_probability:
            xi = proposed_state
            c += 1
        r.append(xi)
    return np.r_[r], c*1./n_samples


def confusion_matrix(true_labels, predicted_labels):
    from sklearn.metrics import confusion_matrix as np_confusion_matrix
    v = true_labels
    p = predicted_labels
    u = np.sort(np.unique(list(np.unique(p))+list(np.unique(v))))
    cm = pd.DataFrame(np_confusion_matrix(v, p))
    cm.index = pd.Index(u, name="true")
    cm.columns = pd.Index(u, name="predicted")
    return cm


def kdensity_smoothed_histogram(x):
    if len(x.shape) != 1:
       raise ValueError("x must be a vector. found "+str(x.shape)+" dimensions")
    x = pd.Series(x).dropna().values
    t = np.linspace(np.min(x), np.max(x), 100)
    p = kdensity(x)(t)
    return t, p

def kdensity(x):
    import numbers
    if len(x.shape) != 1:
        raise ValueError("x must be a vector. found "+str(x.shape)+" dimensions")
    stdx = np.std(x)
    bw = 1.06*stdx*len(x)**-.2 if stdx != 0 else 1.
    kd = KernelDensity(bandwidth=bw)
    kd.fit(x.reshape(-1, 1))

    func = lambda z: np.exp(kd.score_samples(np.array(z).reshape(-1, 1)))
    return func


def plot_1D_mcmc_trace(k, p=None, title=None, x_range=None,
                       acorr_lags=50, burnin=None):
    kb = k if burnin is None else k[burnin:]
    plt.subplot2grid((1, 5), (0, 0), colspan=3)
    plt.plot(range(len(k)), k)
    if burnin is not None:
        plt.plot(range(burnin, len(k)), kb, color="red", alpha=.5)
    if title is not None:
        plt.title(title)
    plt.subplot2grid((1, 5), (0, 3))
    plt.hist(kb, bins=30, alpha=.5, normed=True)
    xr = np.linspace(np.min(kb) if x_range is None else x_range[0],
                     np.max(kb) if x_range is None else x_range[1], 100)
    plt.xlim(np.min(xr), np.max(xr))
    if p is not None:
        plt.plot(xr, p(xr), color="red", lw=2, label="actual probability")
    xt, yt = kdensity_smoothed_histogram(kb)
    plt.plot(xt, yt, color="black", lw=2, label="density estimation")
    plt.legend()
    ax = plt.subplot2grid((1, 5), (0, 4))
    plot_acf(kb, lags=acorr_lags, alpha=1., ax=ax)

def plot_mcmc_vars(k, burnin=None):
    idx=int(burnin) if burnin is not None else 0
    burning = int(burnin) if burnin is not None else burnin
    for i in range(k.shape[1]):
        plt.figure(figsize=(20,2))
        plot_1D_mcmc_trace(k[:,i], burnin=burnin, title="mean %.2f   mean after burning %.2f "%(np.mean(k[:,i]), np.mean(k[idx,i])))

class KDClassifier:

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X,y):
        """
        builds a kernel density estimator for each class
        """
        self.kdes = {}
        for c in np.unique(y):
            self.kdes[c] = KernelDensity(**self.kwargs)
            self.kdes[c].fit(X[y==c])
        return self

    def predict(self, X):
        """
        predicts the class with highest kernel density probability
        """
        classes = self.kdes.keys()
        preds = []
        for i in sorted(classes):
            preds.append(self.kdes[i].score_samples(X))
        preds = np.array(preds).T
        preds = preds.argmax(axis=1)
        preds = np.array([classes[i] for i in preds])
        return preds

    def score(self, X, y):
        return np.mean(y == self.predict(X))


def show_image_mosaic(imgs, labels, figsize=(12, 12), idxs=None):
    from rlx.utils import pbar

    plt.figure(figsize=figsize)
    for labi,lab in pbar()([i for i in enumerate(np.unique(labels))]):
        k = imgs[labels == lab]
        _idxs = idxs[:10] if idxs is not None else np.random.permutation(len(k))[:10]
        for i, idx in enumerate(_idxs):
            if i == 0:
                plt.subplot(10, 11, labi*11+1)
                plt.title("LABEL %d" % lab)
                plt.plot(0, 0)
                plt.axis("off")

            img = k[idx]
            plt.subplot(10, 11, labi*11+i+2)
            plt.imshow(img, cmap=plt.cm.Greys_r)
            plt.axis("off")


class Batches:
    """
    creates batches for a set of arrays. execution examples:

        b = Batches([np.r_[range(10)]], batch_size=3, n_steps=6)
        for i in b.get():
            print i
        ---
        [array([0, 1, 2])]
        [array([3, 4, 5])]
        [array([6, 7, 8])]
        [array([9])]
        [array([0, 1, 2])]
        [array([3, 4, 5])]

        or also:

        for batch in ml.Batches([np.r_[range(10)]], batch_size=3, n_epochs=2).get():
            print batch
        ---
        [array([0, 1, 2])]
        [array([3, 4, 5])]
        [array([6, 7, 8])]
        [array([9])]
        [array([0, 1, 2])]
        [array([3, 4, 5])]
        [array([6, 7, 8])]
        [array([9])]

    shuffle: shuffles data everytime it starts yielding batches
    """
    def __init__(self, arrays, batch_size, n_steps=None, n_epochs=None, shuffle=False):
        assert type(arrays) == list, "arrays must be a list of arrays"
        assert np.std([len(i) for i in arrays]) == 0, "all arrays must be of the same length"
        assert not (n_steps is not None and n_epochs is not None), "cannot set both n_steps and n_epochs"
        assert not (n_steps is None and n_epochs is None), "must set either n_steps or n_epochs"

        self.arrays = arrays
        self.len = len(arrays[0])
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_epochs = n_epochs

        self.steps_per_epoch = self.len/self.batch_size+(1 if self.len%self.batch_size!=0 else 0)
        self.n_steps = n_steps if n_steps is not None else self.steps_per_epoch*self.n_epochs

    def get(self):
        steps_done = 0
        while steps_done < self.n_steps:
            if self.shuffle:
                idxs = np.random.permutation(np.arange(self.len))
                self.arrays = [d[idxs] for d in self.arrays]
            idxs = np.random.permutation(np.arange(self.len)) if self.shuffle else np.arange(self.len)
            for i in range(self.steps_per_epoch):
                batch_start = i*self.batch_size
                batch_end = np.min(((batch_start + self.batch_size), self.len))
                yield [d[batch_start:batch_end] for d in self.arrays]
                steps_done += 1
                if steps_done == self.n_steps:
                    break


def get_vgg(num_classes, num_features=224, pkeep_dropout=0.5):
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.conv import conv_2d, max_pool_2d
    from tflearn.layers.estimator import regression

    tf.reset_default_graph()
    network = input_data(shape=[None, num_features, num_features, 3])
    network = conv_2d(network, 64, 3, activation='relu', scope='conv1_1')
    network = conv_2d(network, 64, 3, activation='relu', scope='conv1_2')
    network = max_pool_2d(network, 2, strides=2, name='maxpool1')

    network = conv_2d(network, 128, 3, activation='relu', scope='conv2_1')
    network = conv_2d(network, 128, 3, activation='relu', scope='conv2_2')
    network = max_pool_2d(network, 2, strides=2, name='maxpool2')

    network = conv_2d(network, 256, 3, activation='relu', scope='conv3_1')
    network = conv_2d(network, 256, 3, activation='relu', scope='conv3_2')
    network = conv_2d(network, 256, 3, activation='relu', scope='conv3_3')
    network = max_pool_2d(network, 2, strides=2, name='maxpool3')

    network = conv_2d(network, 512, 3, activation='relu', scope='conv4_1')
    network = conv_2d(network, 512, 3, activation='relu', scope='conv4_2')
    network = conv_2d(network, 512, 3, activation='relu', scope='conv4_3')
    network = max_pool_2d(network, 2, strides=2, name='maxpool4')

    network = conv_2d(network, 512, 3, activation='relu', scope='conv5_1')
    network = conv_2d(network, 512, 3, activation='relu', scope='conv5_2')
    network = conv_2d(network, 512, 3, activation='relu', scope='conv5_3')
    network = max_pool_2d(network, 2, strides=2, name='maxpool5')

    network = fully_connected(network, 4096, activation='relu', scope='fc6')
    network = dropout(network, pkeep_dropout, name='dropout1')

    network = fully_connected(network, 4096, activation='relu', scope='fc7')
    network = dropout(network, pkeep_dropout, name='dropout2')

    network = fully_connected(network, num_classes, activation='softmax', scope='fc8', restore=False)
    network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.0001)

    return network


def set_vgg_model(model, fname):
    model.load(fname, weights_only=True)
    return model


def get_alexnet(num_classes, pkeep_dropout=.5):
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.conv import conv_2d, max_pool_2d
    from tflearn.layers.normalization import local_response_normalization
    from tflearn.layers.estimator import regression

    # Building 'AlexNet'
    tf.reset_default_graph()
    network = input_data(shape=[None, 227, 227, 3])

    # conv1
    network = conv_2d(network, 96, 11, strides=4, activation='relu', name="conv1", padding="VALID")
    network = max_pool_2d(network, 3, strides=2, name="pool", padding="VALID")
    network = local_response_normalization(network, name="norm1")

    # conv2
    network = conv_2d(network, 256, 5, activation='relu', name="conv2", padding="VALID")
    network = max_pool_2d(network, 3, strides=2, padding="VALID")
    network = local_response_normalization(network)

    # conv3
    network = conv_2d(network, 384, 3, activation='relu', name="conv3")

    # conv4
    network = conv_2d(network, 384, 3, activation='relu', name="conv4")

    # conv5
    network = conv_2d(network, 256, 3, activation='relu', name="conv5")
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    # fc6
    network = fully_connected(network, 4096, activation='tanh', name="fc6")
    network = dropout(network, pkeep_dropout)

    # fc
    network = fully_connected(network, 4096, activation='tanh', name="fc7")
    network = dropout(network, pkeep_dropout)

    network = fully_connected(network, num_classes, activation='softmax', restore=False, name="fc8")
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    return network


"""def download_alexnet_weights(dest_dir="/tmp"):
    url = "https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy"
    dest_file = dest_dir+"/"+url.split("/")[-1]
    wget_cmd = "wget -c %s -o %s"%(url, dest_file)
    !wget -c $url -O $dest_file
    return dest_file
"""


def show_filters(model, layer_name="conv1/W:0"):
    vars = {i.name:i for i in tflearn.variables.get_all_variables()}
    w1 = model.get_weights(vars[layer_name])
    from rlx.utils import pbar
    plt.figure(figsize=(6,6))
    for i in pbar()(range(w1.shape[-1])):
        plt.subplot(10,10,i+1)
        img = w1[:,:,:,i]
        img = (img-np.min(img))/(np.max(img)-np.min(img))
        plt.imshow(img)
        plt.axis("off")


def load_alexnet_weights(fname):
    w = np.load(fname, encoding='bytes').item()
    from rlx.utils import flatten
    weights = {i[0]: i[1] for i in flatten([[[k+"/W:0", w[k][0]], [k+"/b:0", w[k][1]]] for k in w.keys()])}

    # these weights are duplicated (AlexNet has two groups of convolutions for parallelization)
    for k in ["conv2/W:0", "conv4/W:0", "conv5/W:0"]:
        weights[k] = np.concatenate((weights[k], weights[k]), axis=2)
    return weights


def set_alexnet_weights(model, weights, verbose=False):
    import tflearn
    vars = {i.name: i for i in tflearn.variables.get_all_variables()}
    for k in np.sort(weights.keys()):
        if not k.startswith("fc8"):
            if verbose:
                print "setting weights in", k
            model.set_weights(vars[k], weights[k])
        else:
            if verbose:
                print "skipping weights in", k
    return model

def plot_2Ddata_with_boundary(predict, X, y):
    n = 200
    mins, maxs = np.min(X, axis=0), np.max(X, axis=0)
    mins -= np.abs(mins)*.2
    maxs += np.abs(maxs)*.2
    d0 = np.linspace(mins[0], maxs[0], n)
    d1 = np.linspace(mins[1], maxs[1], n)
    gd0, gd1 = np.meshgrid(d0, d1)
    D = np.hstack((gd0.reshape(-1, 1), gd1.reshape(-1, 1)))
    p = (predict(D)*1.).reshape((n, n))
    plt.contourf(gd0, gd1, p, levels=[-0.1, 0.5], alpha=0.5, cmap=plt.cm.Greys)
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c="blue")
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c="red")

def flip_images(X_imgs, show_progress_bar=False):
    from rlx.utils import pbar
    IMAGE_SIZE_1, IMAGE_SIZE_2 = X_imgs.shape[1], X_imgs.shape[2]

    X_flip = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (IMAGE_SIZE_1, IMAGE_SIZE_2, 3))
    tf_img1 = tf.image.flip_left_right(X)
    tf_img2 = tf.image.flip_up_down(X)
    tf_img3 = tf.image.transpose_image(X)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in (pbar()(X_imgs) if show_progress_bar else X_imgs):
            flipped_imgs = sess.run([tf_img1, tf_img2, tf_img3], feed_dict = {X: img})
            X_flip.extend(flipped_imgs)
    X_flip = np.array(X_flip, dtype = np.float32)
    return X_flip

def rotate_images(X_imgs, start_angle, end_angle, n_images, show_progress_bar=False):
    from rlx.utils import pbar, flatten
    IMAGE_SIZE_1, IMAGE_SIZE_2 = X_imgs.shape[1], X_imgs.shape[2]

    X_rotate = []
    iterate_at = (end_angle - start_angle) / (n_images - 1)

    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (None, IMAGE_SIZE_1, IMAGE_SIZE_2, 3))
    radian = tf.placeholder(tf.float32, shape = (len(X_imgs)))
    tf_img = tf.contrib.image.rotate(X, radian)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for index in (pbar()(range(n_images)) if show_progress_bar else range(n_images)):
            degrees_angle = start_angle + index * iterate_at
            radian_value = degrees_angle * np.pi / 180  # Convert to radian
            radian_arr = [radian_value] * len(X_imgs)
            rotated_imgs = sess.run(tf_img, feed_dict = {X: X_imgs, radian: radian_arr})
            X_rotate.extend(rotated_imgs)

    X_rotate = np.array(X_rotate, dtype=np.float32)
    X_rotate = X_rotate[np.r_[flatten([[i, i+len(X_imgs)] for i in np.arange(len(X_imgs))])]]
    return X_rotate


def imread_normalized(fname):
    """
    reads image and normalizes pixel values to the [0,1] interval
    """
    r = imread(fname).astype(np.float32)
    r = (r-np.min(r))/(np.max(r)-np.min(r))
    return r

def scale_images(X_imgs, scales, show_progress_bar=False):
    """
    X_imgs: shape [n_imgs, size_x, size_y, n_channels]
    scales: p.ej.: [.9, .5] produces 2 new images (scale <0 means larger)
    """
    from rlx.utils import pbar
    IMAGE_SIZE_1, IMAGE_SIZE_2 = X_imgs.shape[1], X_imgs.shape[2]
    # Various settings needed for Tensorflow operation
    boxes = np.zeros((len(scales), 4), dtype=np.float32)
    for index, scale in enumerate(scales):
        x1 = y1 = 0.5 - 0.5 * scale  # To scale centrally
        x2 = y2 = 0.5 + 0.5 * scale
        boxes[index] = np.array([y1, x1, y2, x2], dtype=np.float32)
    box_ind = np.zeros((len(scales)), dtype=np.int32)
    crop_size = np.array([IMAGE_SIZE_1, IMAGE_SIZE_2], dtype=np.int32)

    X_scale_data = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=(1, IMAGE_SIZE_1, IMAGE_SIZE_2, 3))
    # Define Tensorflow operation for all scales but only one base image at a time
    tf_img = tf.image.crop_and_resize(X, boxes, box_ind, crop_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for img_data in (pbar()(X_imgs) if show_progress_bar else X_imgs):
            batch_img = np.expand_dims(img_data, axis=0)
            scaled_imgs = sess.run(tf_img, feed_dict={X: batch_img})
            X_scale_data.extend(scaled_imgs)

#    X_scale_data = np.array(X_scale_data, dtype=np.float32)
    return X_scale_data

def augment_imgs(imgs, dataset, prob_augment, op, opname):
    """
    dataset must be indexed with the image file name and must have a field
    named "path" with the full path to the image.
    a one to one correspondance is assumed to between imgs and entries on dataset.

    returns: augmented_imgs
             augmented_dataset: where corresponding records in the dataset
             are copied and paths and indexes appropriated updated.
    """
    assert len(imgs) == len(dataset), "dataset and imgs must have the same length"

    from rlx.utils import pbar
    from skimage.io import imsave

    imgs_to_augment = np.random.permutation(len(imgs))[:int(len(imgs)*prob_augment)]
    print "applying operation", opname
    augmented_imgs = op(imgs[imgs_to_augment])

    r  = len(augmented_imgs)/len(imgs_to_augment)

    augmented_dataset = None

    t = np.random.randint(len(imgs_to_augment))
    print "saving imgs and building dataset"
    for t in pbar()(range(len(imgs_to_augment))):
        z  = dataset.iloc[imgs_to_augment[t]]
        zd = pd.DataFrame([z]*r, index=["%s__%s_%d.jpg"%(z.name,opname, i) for i in range(r)])
        zd["path"] = ["/".join(i.path.split("/")[:-1]+[i.name]) for _,i in zd[["path"]].iterrows()]
        for i,(_,item) in enumerate(zd.iterrows()):
            imsave(item.path, augmented_imgs[r*t+i])
        augmented_dataset = zd if augmented_dataset is None else pd.concat((augmented_dataset, zd))

    return augmented_imgs, augmented_dataset


class RBM:

    """
    variables and methods

    s_*: symbolic expressions (either explicit or obtain by sympy derivation)
    n_*: numeric expressions through the exact formula
    b_*: numeric expressions by brute force (exhaustive)

    """

    def __init__(self, dim_x=3, dim_h=2,
                 compute_exhaustive=False,
                 X_domain = None):
        """
            compute_exhaustive: computes structures for symbolic
            computations, full domain for X and H, exhaustive probabilities,
            etc.
            X_domain: explicit list of all possible inputs distribution
        """
#        assert compute_exhaustive and X_domain is not None, "cannot set both compute_exhaustive and X_domain"
        self.dim_x = dim_x
        self.dim_h = dim_h
        self.X = None
        self.compute_exhaustive = compute_exhaustive
        if compute_exhaustive:
            self.create_domain()
        self.set_random_Wcb()

        self.theta_names = [["W", i[0][0], i[0][1]] for i in np.ndenumerate(self.W)] +\
                           [["c", i[0][0], 0] for i in np.ndenumerate(self.c)] +\
                           [["b", i[0][0], 0] for i in np.ndenumerate(self.b)]

        if compute_exhaustive:
            self.s_create_base_symbols()

        if X_domain is not None:
            self.X = X_domain

    def create_domain(self):
        self.X = np.r_[[i for i in itertools.product(*([[0, 1]]*self.dim_x))]]
        self.H = np.r_[[i for i in itertools.product(*([[0, 1]]*self.dim_h))]]

    def sample_xh(self):
        return self.sample_x(), self.H[np.random.randint(len(self.H))]

    def sample_x(self):
        if self.X is not None:
            return self.X[np.random.randint(len(self.X))]
        else:
            return np.random.randint(2, size=self.dim_x)

    def set_random_Wcb(self):
        W = np.random.normal(size=(self.dim_h, self.dim_x))
        c = np.random.normal(size=self.dim_x)
        b = np.random.normal(size=self.dim_h)
        self.set_Wcb(W, c, b)

    def set_Wcb(self, W=None, c=None, b=None):
        self.W = W if W is not None else self.W
        self.b = b if b is not None else self.b
        self.c = c if c is not None else self.c
        assert self.W is not None and self.c is not None and self.b is not None, "must define W, b and c"
        if self.compute_exhaustive:
            self.compute_probs_exhaustive()

    def compute_probs_exhaustive(self):
        self.b_Z = np.sum([np.exp(-self.n_energy(x, h)) for x, h in itertools.product(self.X,self.H)])
        self.b_probs = np.r_[[[self.n_prob(xi, hi) for xi in self.X] for hi in self.H]].T

    def n_energy(self, x, h):
        return -(h.dot(self.W).dot(x)+self.c.dot(x)+self.b.dot(h))

    def n_prob(self, x, h):
        return np.exp(-self.n_energy(x, h))/self.b_Z

    def sigm(self, x):
        return 1/(1+np.exp(-x))

    def n_prob_h_given_x(self, h, x):
        k = self.sigm(self.W.dot(x)+self.b)
        return np.product(k**h * (1-k)**(1-h))

    def h(self, x, h):
        k = self.sigm(h.dot(self.W)+self.c)
        return np.product(k**x * (1-k)**(1-x))

    def b_prob_x(self, x):
        return self.b_probs.sum(axis=1)[np.alltrue(x==self.X, axis=1)][0]

    def n_free_energy(self, x):
        return -(self.c.dot(x)+np.sum(np.log(1+np.exp(self.W.dot(x)+self.b))))

    def b_prob_h(self, h):
        return self.b_probs.sum(axis=0)[np.alltrue(h == self.H, axis=1)][0]

    def plot_data_probs(self, data, sample_size_domain=1000, title="", figsize=None, sorted=True, show_data_labels=False):
        """
        set sample_size_domain to None if you want to plot the full domain
        """
        if figsize is not None:
            plt.figure(figsize=figsize)

        if sample_size_domain is not None:
            x_samples = np.r_[[self.sample_x() for _ in range(sample_size_domain - len(data))]]
            X = np.r_[list(data)+list(x_samples)]
            idxs_data = np.zeros(len(X)).astype(bool)
            idxs_data[:len(data)]=True
        else:
            X = self.X
            idxs_data = np.r_[[np.sum(np.alltrue(xi==data, axis=1))!=0
                               for xi in self.X]] if data is not None\
                                                  else np.r_[[0]*len(self.X)]


        x_probs     = np.r_[[self.n_free_energy(xi) for xi in X]]

        fe_data   = x_probs[idxs_data]
        fe_domain = x_probs[np.logical_not(idxs_data)]

        sidxs       = np.argsort(-x_probs) if sorted else np.arange(len(x_probs))
        sidxs_data  = idxs_data[sidxs]
        x_probs     = x_probs[sidxs]
        ss = np.r_[[str(i) for i in X]][sidxs]

        plt.plot(x_probs, range(len(x_probs)), color="blue", alpha=.5, label="domain")
        if data is not None:
            plt.scatter(x_probs[sidxs_data], np.argwhere(sidxs_data),
                        color="red", alpha=.5, label="data")
        plt.xlabel("free energy")
        plt.ylabel("RBM input ($x\in \mathbb{R}^{%d}$)"%self.dim_x+(" sorted by free energy" if sorted else ""))
        if show_data_labels:
            plt.yticks(np.arange(len(X)), ss)#, rotation="45", ha="right")
        plt.title(title)

        x,y = kdensity_smoothed_histogram(fe_data)
        plt.plot(x,y*len(X)/np.max(y)*.5, color="red", alpha=.2, ls="--")
        x,y = kdensity_smoothed_histogram(fe_domain)
        plt.plot(x,y*len(X)/np.max(y)*.5, color="blue", alpha=.2, ls="--")
        plt.grid()
        plt.legend()


    def plot_joint_probs(self, figsize=None, show_data_labels=False):
        if figsize is not None:
            plt.figure(figsize=figsize)
        plt.imshow(self.b_probs.T, origin="bottom", cmap=plt.cm.plasma, interpolation="none")
        if show_data_labels:
            plt.yticks(range(len(self.H)), [str(i) for i in self.H])
            plt.xticks(range(len(self.X)), [str(i) for i in self.X], rotation="45", ha="right");
        plt.ylabel("hidden units $h$")
        plt.xlabel("input units $x$")
        plt.colorbar();

    def plot_log_data_energy(self):
        plt.plot([i[0] for i in self.log_data_energy], [i[1] for i in self.log_data_energy], label="data energy")
        plt.ylabel("mean energy of train data")
        plt.xlabel("optimization iteration")
        plt.legend(loc="lower left")
        plt.grid()

    def get_vectorized_params(self):
        """
        gets a vector with al param values in order established by the symbolic keys
        """
        vals = []
        for k in self.theta_names:
            name, i, j = k[0], k[1], k[2]
            vals.append(self.W[i,j] if name=="W" else self.b[i] if name=="b" else self.c[i] if name=="c" else None)
        return vals

    def set_vectorized_params(self, vals):
        for count, k in enumerate(self.theta_names):
            name, i, j = k[0], k[1], k[2]
            if name == "W":
                self.W[i, j] = vals[count]
            elif name == "b":
                self.b[i] = vals[count]
            elif name == "c":
                self.c[i] = vals[count]
        return self.W, self.c.reshape(-1, 1), self.b.reshape(-1, 1)

    def s_create_base_symbols(self):
        self.mW = sy.MatrixSymbol("W", self.dim_h, self.dim_x)
        self.mx = sy.MatrixSymbol("x", self.dim_x, 1)
        self.mc = sy.MatrixSymbol("c", self.dim_x, 1)
        self.mh = sy.MatrixSymbol("h", self.dim_h, 1)
        self.mb = sy.MatrixSymbol("b", self.dim_h, 1)

        self.s_W = sy.Matrix(self.mW)
        self.s_x = sy.Matrix(self.mx)
        self.s_c = sy.Matrix(self.mc)
        self.s_h = sy.Matrix(self.mh)
        self.s_b = sy.Matrix(self.mb)

        self.sykeys = [self.s_W[i[0]] for i in np.ndenumerate(self.W)] + \
                      [self.s_c[i[0]] for i in np.ndenumerate(self.c.reshape(-1, 1))] + \
                      [self.s_b[i[0]] for i in np.ndenumerate(self.b.reshape(-1, 1))]


    def s_create_probability_symbols(self, verbose=False):
        if verbose:
            print "creating SymPy symbols"

        self.s_create_base_symbols()

        self.s_E = -(self.s_h.T*self.s_W*self.s_x+self.s_c.T*self.s_x+self.s_b.T*self.s_h)
        self.s_energy = self.s_E[0]

        self.s_eE = sy.exp(-self.s_E)

        self.s_X = [{self.s_x[i]: xi[i] for i in range(len(xi))} for xi in self.X]
        self.s_H = [{self.s_h[i]: hi[i] for i in range(len(hi))} for hi in self.H]

        self.s_Z = np.sum([self.s_eE.subs(x_).subs(h_)
                        for x_, h_ in itertools.product(self.s_X, self.s_H)])
        self.s_prob = self.s_eE/self.s_Z
        self.s_prob_x = np.sum([self.s_prob.subs(h_) for h_ in self.s_H])

        self.l_Z = sy.lambdify((self.mW, self.mc, self.mb), self.s_Z, "numpy")
        self.l_prob = sy.lambdify((self.mW, self.mc, self.mb, self.mx, self.mh), self.s_prob, "numpy")
        self.l_prob_x = sy.lambdify((self.mW, self.mc, self.mb, self.mx), self.s_prob_x, "numpy")

        self.s_free_energy = -(self.s_c.dot(self.s_x)+np.sum([sy.log(1+sy.exp(self.s_b[j,0]+(self.s_W[j,:].dot(self.s_x)))) for j in range(self.dim_h)]))
        tmpl_free_energy = sy.lambdify((self.mx, self.mW, self.mc, self.mb), self.s_free_energy, "numpy")
        self.l_free_energy = lambda x: tmpl_free_energy(x.reshape(-1,1), self.W, self.c.reshape(-1,1), self.b.reshape(-1,1))

    def n_free_energy_grad(self, xi, wrt):
        if wrt[0]=="W":
            m, n = wrt[1], wrt[2]
            ex = np.exp(self.b[m]+self.W[m, :].dot(xi))
            return -xi[n]*ex/(1+ex)
        elif wrt[0]=="b":
            m = wrt[1]
            ex = np.exp(self.b[m]+self.W[m, :].dot(xi))
            return -ex/(1+ex)
        elif wrt[0]=="c":
            n = wrt[1]
            return -xi[n]
        assert False, "symbol "+str(wrt)+" is not part of model parameters"


    def s_compute_likelihood(self, X, verbose=False):
        if verbose:
            print "building symbolic likelihood expression"
        self.s_create_probability_symbols(verbose)
        dataX_ = [{self.s_x[i]: xi[i] for i in range(len(xi))} for xi in X]

        self.s_log_likelihood = sy.log(np.product([self.s_prob_x.subs(xi) for xi in dataX_]))

        if verbose:
            print "building symbolic likelihood gradient expression"
        self.s_W_grad = {i[1]: self.s_log_likelihood.diff(i[1]) for i in np.ndenumerate(self.s_W)}
        self.s_b_grad = {i[1]: self.s_log_likelihood.diff(i[1]) for i in np.ndenumerate(self.s_b)}
        self.s_c_grad = {i[1]: self.s_log_likelihood.diff(i[1]) for i in np.ndenumerate(self.s_c)}

        self.s_grads = utils.merge_dicts(self.s_W_grad, self.s_b_grad, self.s_c_grad)

        if verbose:
            print "compiling likelihood"
        self.l_log_likelihood = sy.lambdify((self.mW, self.mc, self.mb), self.s_log_likelihood, "numpy")

        if verbose:
            print "compiling likelihood gradient, with", len(self.sykeys), "parameters"
        self.l_log_likelihood_grads = {k: sy.lambdify((self.mW, self.mc, self.mb), self.s_grads[k], "numpy")
                        for k in utils.pbar()(self.sykeys)}


    def fit_symbolic(self, X_train, n_iters=10, verbose=False):
        self.s_compute_likelihood(X_train, verbose)

        if verbose:
            print "gradient descent"
        k = self.get_vectorized_params()

        self.log_data_energy = []

        for it in utils.pbar()(range(n_iters)):
            k += np.r_[[self.l_log_likelihood_grads[i](*self.set_vectorized_params(k)) for i in self.sykeys]]
            self.log_data_energy.append((it, np.mean([self.n_free_energy(xi) for xi in X_train])))

        self.set_vectorized_params(k)
        self.compute_probs_exhaustive()

    def fit_mcmc(self, X_train, n_cycles=1, n_iters=100):
        k = self.get_vectorized_params()

        X1 = X_train
        self.log_data_energy = []

        for it in utils.pbar()(range(n_iters)):
            mgrad, X1 = self.compute_mcmc_gradient(n=n_cycles, X_train=X_train, X_init=X1)
            k -= mgrad
            self.set_vectorized_params(k)
            self.log_data_energy.append((it, np.mean([self.n_free_energy(xi) for xi in X_train])))

    def mcmc_chain(self, n, init_x=None):
        init_x = init_x if init_x is not None else self.sample_x()
        x_samples = [init_x]
        for _ in xrange(n):
            x_proposed = self.sample_x()
            sab = np.exp(+self.n_free_energy(x_samples[-1])-self.n_free_energy(x_proposed))
            accept = np.min((1, sab))
            x_samples.append(x_proposed if np.random.random() < accept else x_samples[-1])
        return np.r_[x_samples[1:]]

    def mcmc_new_dataset(self, n, init_X):
        """
        creates a new dataset from the initial dataset init_X by making an
        MCMC chain starting on n elements from each element of init_X
        keeping the last sample.
        """
        new_dataset = np.r_[[self.mcmc_chain(n, init_x=init_X[i])[-1] for i in xrange(len(init_X))]]
        return new_dataset

    def compute_mcmc_gradient(self, n, X_train, X_init):
        """
        X_train: train data for computing expectation.
        X_init: an mcmc chain will be started from each point in X_init.
        """
        egrad=[]
        X1 = self.mcmc_new_dataset(n=n, init_X=X_init)
        for wrt in self.theta_names:
            e0 = np.mean([self.n_free_energy_grad(xi, wrt) for xi in X_train])
            e1 = np.mean([self.n_free_energy_grad(xi, wrt) for xi in X1])
            egrad.append(e0-e1)
        return np.r_[egrad], X1

    def contrastive_divergence_experiment(self, len_Xtr, n_iters=100,
                                          sample_size=1000, n_cycles=1):
        Xtr = np.random.randint(2, size=(len_Xtr, self.dim_x))
        plt.figure(figsize=(15,3))
        plt.subplot(131)
        self.set_random_Wcb()
        self.plot_data_probs(Xtr, sample_size_domain=np.min((sample_size, 2**self.dim_x)),
                             sorted=True, title="before optimization")

        data_energy = self.fit_mcmc(Xtr, n_cycles=n_cycles, n_iters=n_iters)

        plt.subplot(132)
        self.plot_data_probs(Xtr,  sample_size_domain=np.min((1000, 2**self.dim_x)),
                             sorted=True, title="after optimization")

        plt.subplot(133)
        plt.plot([i[0] for i in self.log_data_energy], [i[1] for i in self.log_data_energy], label="data energy")
        plt.ylabel("mean energy")
        plt.xlabel("optimization iteration")
        plt.legend(loc="lower left")
        plt.grid()

class Segmentation_Label_Map:

    def __init__(self, ref_geodf=None):
        self.init_labelmap()
        self.ref_geodf=ref_geodf
        self.current_intersection = None
        self.current_tile = None

    def set_ref_geodf(self, ref_geodf):
        self.ref_geodf = ref_geodf

    def get_default_rgb(self):
        return (1.,1.,1.)

    def init_labelmap(self):
        raise NotImplementedError

        """
        subclasses must implement this method so that self.labelmap is a Pandas DataFrame
        with a label definition for each row such as in the following example implementation.
        - RGB colors MUST be triplets with values between 0 and 1
        - mandatory columns are "rgb" and "name", the rest of the columns are to be used
          in other methods in the same sub class
        """
        cat_cols = [[255,   0,   0], # red
                    [255,   0, 255], # magenta
                    [255, 128, 64],  # orange
                    [  0,   0, 255], # blue
                    [  0, 255,   0], # green
                    [  0, 128, 255], # cyan
                    [255, 255, 255], # white
                    [  0,   0,   0]] # black

        cat_names = ["constr sobre", "constr bajo", "zona deport", "piscina/estanque",
                     "solar/patio", "parcela rustica", "sin parcela", "unknown"]
        cat_codes = [11,12,13,14,15,16,0,-1]

        self.labelmap = pd.DataFrame([cat_cols, cat_codes, cat_names], index=pd.Index(["rgb", "code", "name"])).T
        self.labelmap["rgb"] = [np.r_[i]/255. if np.max(i)>1 else i for i in self.labelmap.rgb]

    def geometry_label(self, gs):
        """
        abstract method
        gs: a pandas GeoSeries
        returns: an index representing one row of the labelmap dataframe
        """
        raise NotImplementedError

    def create_contrast_image(self):
        im, si = 3,4
        w = (im+si)*4+im
        h = (im+si)*int(np.ceil(len(self.labelmap)/4.))+im
        k = np.zeros((h,w*len(self.labelmap),3))
        for i in range(len(self.labelmap)):
            k[:,i*w:(i+1)*w] = self.labelmap.iloc[i].rgb
            y,x = im, im+i*w
            for j in range(len(self.labelmap)):
                k[y:y+si, x:x+si] = self.labelmap.iloc[j].rgb
                x+=si+im
                if x>=(i+1)*w:
                    y+=im+si
                    x=im+i*w
            
        return k

    def show_labels(self):
        img = self.create_contrast_image()
        k = geo.SampleTile(img)
        plt.figure(figsize=(20,2*np.ceil(len(self.labelmap)/4.)))
        plt.subplot(211)
        plt.imshow(k.get_img())
        plt.title("rgb labels")
        plt.axis("off")
        plt.subplot(212)
        plt.imshow(self.get_labels_from_tile(k).get_img())
        plt.title("single channel labels")
        plt.axis("off")

    def set_tile(self, tile, show_progress=True):
        assert self.ref_geodf is not None, "must set reference geodataframe"
        self.current_tile = tile
        self.current_intersection = tile.intersection(self.ref_geodf, show_progress=show_progress)

    def apply_to_tile(self, geometries_alpha=0.5, show_tile=True, tile_alpha=1., 
                      default_alpha=1., **kwargs):
        """
        embeds geometries from ref_geodf into a single image
        tile: the tile to apply to, if None will use last tile
        returns: an image with the geometries embedded
        """
        assert self.current_tile is not None, "must call 'set_tile' before"

        tile = self.current_tile
        gf   = self.current_intersection

        default_color=self.get_default_rgb()

        # first scale geometries to image pixel range
        pol = tile.get_polygon()
        xmin, ymin = np.array(pol.exterior.xy).min(axis=1)
        xmax, ymax = np.array(pol.exterior.xy).max(axis=1)

        if len(gf)==0:
            kpol = sh.affinity.translate(pol, xoff=-xmin, yoff=-ymin)
            kpol = sh.affinity.scale(kpol, xfact=tile.w*1./(xmax-xmin), yfact=tile.h*1./(ymax-ymin), origin=(0,0))
            pols = [kpol]
            colors = [default_color]

        else:
            pols   = []
            colors = []
            for c,i in enumerate(gf.geometry):
                kpol = sh.affinity.translate(i, xoff=-xmin, yoff=-ymin)
                kpol = sh.affinity.scale(kpol, xfact=tile.w*1./(xmax-xmin), yfact=tile.h*1./(ymax-ymin), origin=(0,0))
                pols.append(kpol)
                colors.append(self.geometry_label(gf.iloc[c]).rgb)

        ki = gpd.GeoSeries(pols)
        minx, miny, maxx, maxy = ki.total_bounds

        fig = plt.figure(figsize=(tile.w*1./100, tile.h*1./100), dpi=100., frameon=True)
        fig.set_facecolor('none')
        ax = plt.gca()

        # paint image
        if show_tile:
#            nimg = np.flip(np.array(tile.get_img().convert("RGB")), axis=0)
            ax.imshow(np.flip(tile.get_img(), axis=0), alpha=tile_alpha)

        # make background polygon
        bpol = sh.geometry.Polygon(([0,0], [tile.w,0], [tile.w,tile.h], [0,tile.h]))

        # paint allpolygons
        for i in range(len(pols)):
            kpol = pols[i]
            ax.add_patch(descartes.PolygonPatch(kpol, color=colors[i],
                                                lw=1, alpha=geometries_alpha))
            bpol = bpol.difference(kpol)

        ## add remaining space with default color
        if bpol.area>0:
            ax.add_patch(descartes.PolygonPatch(bpol, color=default_color, lw=0, alpha=default_alpha))


        plt.axis("off")
        plt.margins(0,0)
        plt.xlim(0,tile.w)
        plt.ylim(0,tile.h)
        plt.subplots_adjust(left=0, right=1., top=1., bottom=0)
        fig.canvas.draw()
        image = Image.fromarray(np.array(fig.canvas.renderer._renderer))
        image = image.resize(image.size, Image.ANTIALIAS)
        image = np.array(image.convert("RGB"))
        r = np.zeros(image.shape).astype("uint8")
        for i in range(3):      
            r[:,:,i] = enhance_contrast_percentile(image[:,:,i], disk(2), p0=.1, p1=.9)

        plt.close()

        r = geo.GeneratedTile.from_image(r, tile, **kwargs)

        return r

    def get_labels_as_rgb(self):
        assert self.current_tile is not None, "must call 'set_tile' before"
        return self.apply_to_tile(show_tile=False, geometries_alpha=1.)

    def get_labels(self):
        assert self.current_tile is not None, "must call 'set_tile' before"
        r = self.get_labels_as_rgb()
        r = self.get_labels_from_tile(r)
        r.format = "png"
        return r

    def get_labels_from_tile(self, tile):
        """
        method to get labels directly from tile without intersecting geometry
        """
        r = tile.get_img()
        r = (np.r_[[np.abs(r-i).sum(axis=2) for i in self.labelmap.rgb.values]].argmin(axis=0)).astype(np.uint8)
        r = geo.GeneratedTile.from_image(r, tile)
        return r

