
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import tflearn
import tensorflow as tf
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.recurrent import lstm
from tflearn.layers.estimator import regression
import rlx.utils as rxu

def generate_model_signal(n, p, noise=0.1, feature_range=(-.3, .3)):
    x = np.linspace(1,50,n)
    s = np.sin(x)*np.cos(x*p[0])*np.sin(p[1]*np.log(x))
    s += np.random.normal(size=s.shape)*noise

    # bound signal around 0
    from sklearn.preprocessing import MinMaxScaler
    s = MinMaxScaler(feature_range=feature_range).fit_transform(s.reshape(-1, 1))[:, 0]

    return s


def build_sequences(signal, seq_len):
    result = []
    for index in range(len(signal) - seq_len):
        result.append(signal[index: index + seq_len])
    result = np.array(result)
    return result


def build_train_test_sequences(s, seq_len, train_pct):
    X = build_sequences(s, seq_len)

    t = int(len(s)*train_pct)

    y = X[:, -1]
    X = X[:, :-1]
    Xtr, ytr = X[:t], y[:t]
    Xts, yts = X[t:], y[t:]

    return Xtr, Xts, ytr, yts

def LR_timeseries_prediction_experiment(signal_params, n_points=500, noise_levels = [0,.1,.2,.3], t=200):
    for noise in noise_levels:
        Xtr, Xts, ytr, yts = build_train_test_sequences(generate_model_signal(n_points,
                                                                              signal_params,
                                                                              noise=noise), seq_len=3, t=t)

        lr = LinearRegression()
        lr.fit(Xtr, ytr)
        tit = "noise %.2f"%noise
        tit += "      train error %.4f"%mean_squared_error(ytr, lr.predict(Xtr))
        tit += "      test error  %.4f"%mean_squared_error(yts, lr.predict(Xts))
        tit += "      lr intercept %.4f"%lr.intercept_
        tit += "      lr coefs [" + ", ".join(["%.4f"%i for i in lr.coef_])+"]"
        plt.figure()
        y = np.concatenate((ytr, yts))
        X = np.vstack((Xtr, Xts))
        plt.plot(y, label="signal",lw=7, alpha=.2)
        plt.plot(lr.predict(X), label="predicted")
        plt.axvline(t, color="black", label="train|test boundary")
        plt.legend();
        plt.title(tit)


class HandMade_RNN:
    def __init__(self, unroll_timesteps, state_size,
                 init_state_at_each_epoch=True,
                 shuffle_data_at_each_epoch=True,
                 reuse_last_training_state=False,
                 num_epochs=10000, verbose=True):

        self.unroll_timesteps = unroll_timesteps
        self.state_size = state_size
        self.shuffle_data_at_each_epoch = shuffle_data_at_each_epoch
        self.init_state_at_each_epoch = init_state_at_each_epoch
        self.num_epochs = num_epochs
        self.reuse_last_training_state = reuse_last_training_state
        self.verbose = verbose

    def fit(self, Xtr, ytr):

        self.lookback_size = Xtr.shape[1]

        ## TF placeholders for data
        if self.verbose:
            print "building placeholders for data"
        self.lookbackX_placeholder = tf.placeholder(tf.float32, [None, self.lookback_size, self.unroll_timesteps])
        self.Y_placeholder = tf.placeholder(tf.float32, [None, 1])

        self.init_state = tf.placeholder(tf.float32, [1, self.state_size])

        # Unpack columns
        inputs_series = tf.unstack(self.lookbackX_placeholder, axis=2)
        ## model params
        if self.verbose:
            print "building params"
        self.W1x = tf.Variable(np.random.rand(self.lookback_size, self.state_size), dtype=tf.float32)
        self.W1s = tf.Variable(np.random.rand(self.state_size, self.state_size), dtype=tf.float32)
        self.b1 = tf.Variable(np.zeros((1, self.state_size)), dtype=tf.float32)

        self.W2 = tf.Variable(np.random.rand(self.state_size, 1), dtype=tf.float32)
        self.b2 = tf.Variable(np.zeros(1), dtype=tf.float32)

        # computational graph
        if self.verbose:
            print "building computation graph"
        current_state = self.init_state
        current_input = inputs_series[0]

        for current_input in inputs_series:
            current_state = tf.tanh(tf.matmul(current_input, self.W1x) + tf.matmul(current_state, self.W1s) + self.b1)

        self.prediction = tf.matmul(current_state, self.W2) + self.b2
        total_loss = tf.reduce_mean(tf.pow(self.prediction - self.Y_placeholder, 2))
        train_step = tf.train.AdagradOptimizer(0.5).minimize(total_loss)

        # train model
        if self.verbose:
            print "training model"
        self.loss = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.num_epochs):

                if self.init_state_at_each_epoch or epoch == 0:
                    _current_state = np.zeros((1, self.state_size))
                else:
                    _current_state = _current_state[-1].reshape(1, -1)
                    _current_state = _current_state / np.linalg.norm(_current_state)

                x = np.r_[[Xtr[i:i + self.unroll_timesteps].T for i in range(len(Xtr) - self.unroll_timesteps)]]
                y = ytr[self.unroll_timesteps:].reshape(-1, 1)

                # shuffle data before feeding it
                if self.shuffle_data_at_each_epoch:
                    idxs = np.random.permutation(range(len(x)))
                    x = x[idxs]
                    y = y[idxs]

                _total_loss, _train_step, _current_state, _prediction = sess.run(
                    [total_loss, train_step, current_state, self.prediction],
                    feed_dict={
                        self.lookbackX_placeholder: x,
                        self.Y_placeholder: y,
                        self.init_state: _current_state
                    })
                self.loss.append(_total_loss)
                if self.verbose and epoch % (self.num_epochs / 10) == 0:
                    print epoch, _total_loss
            print epoch, _total_loss
            self.vW1x, self.vW1s, self.vb1, self.vW2, self.vb2 = sess.run(
                [self.W1x, self.W1s, self.b1, self.W2, self.b2])
            self.train_state = _current_state[-1].reshape(1, -1)
            self.train_state = self.train_state

    def predict(self, Xts):
        state = self.train_state if self.reuse_last_training_state else np.zeros((1, self.state_size))

        with tf.Session() as sess:
            x = np.r_[[Xts[i:i + self.unroll_timesteps].T for i in range(len(Xts) - self.unroll_timesteps)]]
            _prediction = sess.run(
                [self.prediction],
                feed_dict={
                    self.lookbackX_placeholder: x,
                    self.init_state: state,
                    self.W1x: self.vW1x, self.W1s: self.vW1s, self.b1: self.vb1,
                    self.W2: self.vW2, self.b2: self.vb2
                })
            return _prediction[0]

def LR_RNN_comparison(noise, seq_len, rnn_unroll_timesteps, rnn_state_size, rnn_epochs, signal_params):

    Xtr, Xts, ytr, yts = build_train_test_sequences(generate_model_signal(1000, signal_params, noise=noise),
                                                    seq_len=seq_len, t=400)
    rnn = HandMade_RNN(unroll_timesteps=rnn_unroll_timesteps, state_size=rnn_state_size,
                       num_epochs=rnn_epochs)
    rnn.fit(Xtr, ytr)

    lr = LinearRegression()
    lr.fit(Xtr, ytr)

    lr_preds  = lr.predict(Xts[rnn.unroll_timesteps:])
    rnn_preds = rnn.predict(Xts)[:,0]

    y = yts[rnn.unroll_timesteps:]

    tit = "RNN_err=%.4f"%np.sqrt(mean_squared_error(y, rnn_preds)) + ", LR_err=%.4f"%np.sqrt(mean_squared_error(y, lr_preds))
    plt.plot(y, label="signal",lw=7, alpha=.2, color="black")
    plt.plot(rnn_preds, label="RNN", color="red", alpha=.7)
    plt.plot(lr_preds, label="LR", color="blue", alpha=.7)
    plt.legend();
    plt.title(tit)
    return Xtr, Xts, ytr, yts, rnn, lr


def getLSTM(num_features, units, dropout):
    network = input_data([None, 1, num_features])

    for i, u in enumerate(units):
        network = lstm(network, u, dropout=dropout, return_seq=i != len(units) - 1)
    network = fully_connected(network, 10, activation="tanh")  # this defaults to linear activation
    network = fully_connected(network, 1, activation=None)  # this defaults to linear activation
    network = regression(network, optimizer='rmsprop', learning_rate=0.01, loss='mean_square')
    return network


#def lstm_experiment(units, dropout, noise, seq_len, n_epoch, params, signal_len=500, train_pct=.4,
#                    logdir="/tmp/g"):
def lstm_experiment(units, dropout, seq_len, n_epoch, gensignal_func=generate_model_signal,
                    gensignal_params={"n": 500, "p": [.1,1.], "noise": 0},
                    train_pct=.4, logdir="/tmp/g"):

    with rxu.printoptions(formatter={'float': '{: .2f}'.format}):
        exp_name = "units=" + rxu.np2str(units) + "_dropout=%.2f" % dropout + \
                   "_signal=" + gensignal_func.__name__+"::"+str(gensignal_params)+ "_seqlen=%d" % seq_len + "_nepoch=%d" % n_epoch

    print exp_name
    signal = gensignal_func(**gensignal_params)
    Xtr, Xts, ytr, yts = build_train_test_sequences(signal, seq_len=seq_len, train_pct=train_pct)
    train_x = Xtr.reshape(-1, 1, Xtr.shape[1])
    train_y = ytr.reshape(-1, 1)
    test_x = Xts.reshape(-1, 1, Xts.shape[1])
    test_y = yts.reshape(-1, 1)

    x = np.vstack((Xtr, Xts)).reshape(-1, 1, Xts.shape[1])
    y = np.hstack((ytr, yts))

    tf.reset_default_graph()

    mylstm = getLSTM(Xtr.shape[1], units=[10, 20], dropout=.4)
    model_lstm = tflearn.DNN(mylstm, tensorboard_verbose=1, tensorboard_dir=logdir)
    model_lstm.fit(train_x, ytr.reshape(-1, 1), n_epoch=n_epoch, show_metric=True, batch_size=4, run_id=exp_name)

    preds = model_lstm.predict(x)
    plt.plot(preds, label="prediction")
    plt.plot(y, label="signal", lw=7, alpha=.2, color="black")
    plt.title("mean abs err %.4f" % (np.mean(np.abs((y - preds[:, 0])))))
    plt.axvline(len(Xtr), color="black", alpha=.7, label="train | test")
    plt.legend()
    return {"preds": preds, "x": x, "y": y, "model": model_lstm}
