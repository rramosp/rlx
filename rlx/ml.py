import numpy as np
import pandas as pd
from rlx import utils
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from joblib import delayed
from sklearn.neighbors import KernelDensity
from statsmodels.graphics.tsaplots import plot_acf
import itertools
import sympy as sy


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

def kdensity_smoothed_histogram(x):
    if len(x.shape)!=1:
       raise ValueError("x must be a vector. found "+str(x.shape)+" dimensions")
    x = pd.Series(x).dropna().values
    t = np.linspace(np.min(x), np.max(x),100)
    p = kdensity(x)(t)
    return t,p

def kdensity(x):
    import numbers
    if len(x.shape)!=1:
       raise ValueError("x must be a vector. found "+str(x.shape)+" dimensions")
    stdx = np.std(x)
    bw = 1.06*stdx*len(x)**-.2 if stdx!=0 else 1.
    kd = KernelDensity(bandwidth=bw)
    kd.fit(x.reshape(-1,1))

    func = lambda z: np.exp(kd.score_samples(np.array(z).reshape(-1,1)))
    return func


def plot_1D_mcmc_trace(k, p=None, title=None, x_range=None, acorr_lags=50, burnin=None):
    kb     = k if burnin is None else k[burnin:]
    plt.subplot2grid((1, 5), (0, 0), colspan=3)
    plt.plot(range(len(k)), k)
    if burnin is not None:
        plt.plot(range(burnin, len(k)), kb, color="red", alpha=.5)
    if title is not None:
        plt.title(title)
    plt.subplot2grid((1, 5), (0, 3))
    plt.hist(kb, bins=30, alpha=.5, normed=True);
    xr = np.linspace(np.min(kb) if x_range is None else x_range[0],
                     np.max(kb) if x_range is None else x_range[1],100)
    plt.xlim(np.min(xr), np.max(xr))
    if p is not None:
        plt.plot(xr, p(xr), color="red", lw=2, label="actual probability")
    xt,yt = kdensity_smoothed_histogram(kb)
    plt.plot(xt,yt,color="black", lw=2, label="density estimation")
    plt.legend()
    ax=plt.subplot2grid((1,5),(0,4))
    plot_acf(kb, lags=acorr_lags, alpha=1.,ax=ax);

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

class Batches:
    def __init__(self, arrays, batch_size, shuffle=False):
        assert type(arrays)==list, "arrays must be a list of arrays"
        assert np.std([len(i) for i in arrays])==0, "all arrays must be of the same length"

        self.arrays = arrays
        self.len = len(arrays[0])
        self.batch_size = batch_size
        self.idxs = np.random.permutation(np.arange(self.len)) if shuffle else np.arange(self.len)

    def get(self):
        for i in range(self.len/self.batch_size+(1 if self.len%self.batch_size!=0 else 0) ):
            batch_start = i*self.batch_size
            batch_end   = np.min(((batch_start + self.batch_size) , self.len))
            yield [d[self.idxs][batch_start:batch_end] for d in self.arrays]


def plot_2Ddata_with_boundary(predict,X,y):
    n = 200
    mins,maxs = np.min(X,axis=0), np.max(X,axis=0)
    mins -= np.abs(mins)*.2
    maxs += np.abs(maxs)*.2
    d0 = np.linspace(mins[0], maxs[0],n)
    d1 = np.linspace(mins[1], maxs[1],n)
    gd0,gd1 = np.meshgrid(d0,d1)
    D = np.hstack((gd0.reshape(-1,1), gd1.reshape(-1,1)))
    p = (predict(D)*1.).reshape((n,n))
    plt.contourf(gd0,gd1,p, levels=[-0.1,0.5], alpha=0.5, cmap=plt.cm.Greys)
    plt.scatter(X[y==0][:,0], X[y==0][:,1], c="blue")
    plt.scatter(X[y==1][:,0], X[y==1][:,1], c="red")


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
