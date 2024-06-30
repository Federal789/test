<<<<<<< HEAD
# coding: utf-8

import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
np.random.seed(1)
import tensorflow as tf
from tfdeterminism import patch
patch()
tf.compat.v1.set_random_seed(1)

import random
random.seed(1)
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONHASHSEED"] = "1"

plt.rcParams['font.sans-serif'] = ['Arial']    # 中文
plt.rcParams['axes.unicode_minus'] = False

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

import keras
from keras import backend as K
from keras import initializers
import keras_metrics as km

from keras.layers import (Activation, BatchNormalization, Dense,
                          Dropout, Flatten, Lambda, Reshape)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model

from sklearn import preprocessing, neighbors
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import *
from sklearn.metrics import *
from sklearn.manifold import TSNE
from keras.optimizers import Adam

from scipy.stats import yeojohnson

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
K.set_session(sess)
#

# Well E
# 1% label ratio
batch_size = 16
para_size = 25      # The number of the logging samples in one segment.
channels = 1
rho = 0.8           # The degree of class balance
z_dim = 60          # The dimension of noise
lr_1 = 0.0002       # Generator
lr_2 = 0.0002       # Discriminator
alpha = 0.01        # The parameter of LeakyReLU
gamma = 1           # The degree of weight reduction
dropout = 0.5
beta1 = 0.9         # Parameter of Adam Optimizer
per = 1             # Label percentage
mt_size = 25        # Same as para_size

# 3% label ratio
# batch_size = 32
# para_size = 25
# channels = 1
# rho = 0.5
# z_dim = 50  # 50
# lr_1 = 0.0005
# lr_2 = 0.0002
# alpha = 0.01
# gamma = 1
# dropout = 0.5
# beta1 = 0.9
# per = 3
# mt_size = 25

# 5% label ratio
# batch_size = 64
# para_size = 25
# channels = 1
# rho = 0.5
# z_dim = 60
# lr_1 = 0.0005
# lr_2 = 0.0001
# alpha = 0.01
# gamma = 1
# dropout = 0.5
# beta1 = 0.9
# per = 5
# mt_size = 25

# 9% label ratio
# batch_size = 128
# para_size = 25
# channels = 1
# rho = 0.8
# z_dim = 60
# lr_1 = 2e-4
# lr_2 = 2e-4
# alpha = 0.01
# gamma = 1
# dropout = 0.5
# beta1 = 0.9
# per = 9
# mt_size = 25

# 7% label ratio
# batch_size = 64
# para_size = 25
# channels = 1
# rho = 0.6
# z_dim = 85
# lr_1 = 2e-4
# lr_2 = 2e-4
# alpha = 0.01
# gamma = 1
# dropout = 0.5
# beta1 = 0.9
# per = 7
# mt_size = 25

# Well A
# batch_size = 32
# para_size = 25
# channels = 1
# rho = 0.5
# z_dim = 45
# lr_1 = 2e-4
# lr_2 = 2e-4
# alpha = 0.01
# gamma = 1
# dropout = 0.3
# beta1 = 0.9
# per = 5
# mt_size = 25

# Well B
# batch_size = 64
# para_size = 25
# channels = 1
# rho = 0.8
# z_dim = 50
# lr_1 = 4e-4
# lr_2 = 2e-4
# alpha = 0.01
# gamma = 1
# dropout = 0.2
# beta1 = 0.9
# per = 5
# mt_size = 25

# Well C
# batch_size = 64
# para_size = 25
# channels = 1
# rho = 0.5
# z_dim = 60
# lr_1 = 0.0004
# lr_2 = 0.0002
# alpha = 0.01
# gamma = 1
# dropout = 0.5
# beta1 = 0.9
# per = 5
# mt_size = 25

# Well D
# batch_size = 128
# para_size = 25
# channels = 1
# rho = 0.8
# z_dim = 60
# lr_1 = 0.0003
# lr_2 = 0.00015
# alpha = 0.01
# gamma = 1
# dropout = 0.5
# beta1 = 0.9
# per = 5
# mt_size = 25


#
def scan_file(path):
    d = []
    for root, dirs, files in os.walk(path):
        d.append(root)

    return d[1: len(d)]

filelist_com = scan_file('images/comparison')
del_root = [i for i in filelist_com if len(i) <= len(filelist_com[0])]
for dr in del_root:
    filelist_com.remove(dr)

path0 = r'3.xlsx'

def loaddata(path):
    data = pd.read_excel(path).values
    Y0 = data[:, -1]
    X0 = data[:, : -1]
    Y0[np.where(Y0 == 4)[0]] = 3
    Y = Y0.copy()
    X_ = X0.copy()
    mm_tool = StandardScaler()
    X = mm_tool.fit_transform(X_)
    return X0, X, Y, X0[:, 0]

SAM_R, SAM, LITHO, DEP = loaddata(path0)
encoder = LabelEncoder()

def get_order(Y):
    Y_sel_1, Y_sel_2 = Y[1: -1], Y[0: -2]
    location1 = np.where(Y_sel_1 != Y_sel_2)[0]
    location = location1[::-1]
    l1 = [len(Y_sel_1) - location[0]]
    l2 = []
    l3 = [location[-1]]
    for x in range(len(location)-1):
        l2.append(location[x] - location[x+1])
    location0 = np.array(l1 + l2 + l3)
    Y_sel_ = encoder.fit_transform(Y_sel_2)
    Y_sel__ = encoder.fit_transform(Y_sel_1)
    Y_sel_2 = Y_sel_.copy()
    Y_sel_1 = Y_sel__.copy()
    return location, location0, Y_sel_1, Y_sel_2

num_samples_ori, num_features = np.shape(SAM)  # 数据集中类别的数量

X = SAM.reshape(len(LITHO), num_features)
B = np.zeros(len(LITHO))
max_X, min_X = [], []
for i in range(num_features):
    B = X[:, i]
    max_X.append(np.max(B))
    min_X.append(np.min(B))

Y = LITHO.copy()

sam_shape = (para_size, num_features, channels)


def getRecall(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))                  #TP
    P=K.sum(K.round(K.clip(y_true, 0, 1)))
    FN = P-TP #FN=P-TP
    recall = TP / (TP + FN + K.epsilon())#TP/(TP+FN)
    return recall

def getPrecision(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))                  #TP
    N = (-1)*K.sum(K.round(K.clip(y_true-K.ones_like(y_true), -1, 0)))   #N
    TN=K.sum(K.round(K.clip((y_true-K.ones_like(y_true))*(y_pred-K.ones_like(y_pred)), 0, 1)))#TN
    FP=N-TN
    precision = TP / (TP + FP + K.epsilon())#TT/P
    return precision

def getf1(y_true, y_pred):
    prec = getPrecision(y_true, y_pred)
    rec = getRecall(y_true, y_pred)
    f1 = 2 * prec * rec / (prec + rec)
    return f1

def plot_embedding_disc(result, y_train, y_test, title, per, name, N=.1, M=.1):
    nor_tool = MinMaxScaler(feature_range=[0, 1])
    data = nor_tool.fit_transform(result)
    x_train, x_test = data[: len(y_train), :], data[len(y_train): np.shape(result)[0], :]
    x1_min, x2_min = x_test.min(axis=0)
    x1_max, x2_max = x_test.max(axis=0)
    print(x1_min, x2_min, x1_max, x2_max)

    t1, t2 = np.meshgrid(np.arange(x1_min, x1_max, N), np.arange(x2_min, x2_max, M))
    x1, x2 = np.meshgrid(t1, t2)
    mesh_input = np.c_[x1.ravel(), x2.ravel()]

    plt.figure()  # 创建一个画布
    markers = ['o', '^', 'D']
    labels = ['RS', 'OS', 'SA']
    colors = ['r', 'g', 'b']
    # plt.pcolormesh(x1, x2, np.array(y_predict).reshape(x1.shape), cmap=cm_light)
    for marker, lab, col in zip(markers, np.unique(y_test), colors):
        plt.scatter(x_test[:, 0].real[y_test == lab],
                    x_test[:, 1].real[y_test == lab],
                    marker=marker, s=15,
                    alpha=0.5, label=labels[int(lab)],
                    edgecolors='k', )
    plt.legend(loc='upper right', borderpad=0.5, fancybox=True)

    plt.title(title, fontdict={'weight': 'heavy', 'size': 14})
    plt.savefig('./images/tsne/per{}'.format(per)+'/disc'+'/{}.jpg'.format(name))


def plot_embedding_gen(result, label, num, title, per, name, N=100, M=100):
    nor_tool = MinMaxScaler(feature_range=[0, 1])
    data = nor_tool.fit_transform(result)

    plt.figure(dpi=500)
    markers = ['o', '^', 's', 'o', '^', 's', 'x']
    labels = ['Real-RS', 'Real-OS', 'Real-SA', 'Real-unlabelled', 'Real-unlabelled', 'Real-unlabelled', 'Fake']
    colors = ['#CD2626', '#228B22', '#4682B4', '#CD2626', '#228B22', '#4682B4', '#6A5ACD']
    edges = ['k', 'k', 'k', None, None, None, None]
    for lab in np.unique(label):
        plt.scatter(data[:, 0].real[label == lab],
                    data[:, 1].real[label == lab],
                    marker=markers[int(lab)], s=25,
                    alpha=0.8, label=labels[int(lab)],
                    color=colors[int(lab)],
                    edgecolors=edges[int(lab)],
                    linewidths=1.5)
    plt.legend(loc='best', prop={'size': 13}, borderpad=0.5, fancybox=True)

    plt.title(title, fontdict={'weight': 600, 'size': 19.5})  # 设置标题
    plt.savefig('./images/tsne/per{}'.format(per)+'/gen/gen_{}'.format(num)+'/{}.jpg'.format(name))


def BUILD_WEIGHT_MAT(Y,  # a label matrix with shape (n,1) or (n,)
                     rho=0.,  # Balance the rare class
                     additional_weights=[],  # additional weights for some samples
                     show_report=True  # Show report if true
                     ):
    # X0: Get basic information
    set_class, num_per_class = np.unique(Y, return_counts=True)  # set of classes and number per class
    num_sample = len(Y)

    if -1 in set_class:
        weight_per_class = 1.0 / (num_per_class ** rho)
        weight_per_class[np.where(set_class == -1)[0]] = 0
    else:
        weight_per_class = 1.0 / (num_per_class ** rho)

    weight_per_class = weight_per_class / sum(weight_per_class)

    if show_report:
        print('# Classes:', set_class, '\n')
        print('# Numbers:', num_per_class, '\n')
        print('# Weights:', weight_per_class, '\n')

    # Build weight_matrix 'C'
    tv = np.zeros(num_sample)  # diagonal elements for 'C'
    for i in set_class:
        tv[list(np.where(Y == i)[0])] = weight_per_class[np.where(set_class == i)[0]]
    if len(additional_weights) > 0:
        tv = tv * additional_weights
    C = tv

    return set_class, C, weight_per_class, num_per_class

SKF = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)

class Dataset:
    def __init__(self, para_size, num_samples_ori, num_features, batch_size, per):
        self.p = para_size
        self.c = num_features
        self.n = num_samples_ori
        self.b = batch_size
        self.per = per

        A1 = []

        for i in range(self.n):
            if (i >= (int((self.p - 1) / 2))) & (i <= int((self.n - (self.p - 1) / 2) - 1)):
                idx1, idx2 = int((i - (self.p - 1) / 2)), int((i + (self.p - 1) / 2 + 1))
                A0 = X[idx1: idx2, :]
            elif i < (int((self.p - 1) / 2)):
                idx1, idx2 = int(((self.p - 1) / 2) + 1), int((i + (self.p - 1) / 2 + 1))
                A0 = np.vstack((X[int(i + 1): idx1, :], X[0: idx2, :]))
            else:
                idx1, idx2 = int((i - (self.p - 1) / 2)), int(self.n - (self.p - 1) / 2 - 1)
                A0 = np.vstack((X[idx1:, :], X[idx2: int(i), :]))

            A_ = A0.reshape(self.p, self.c, 1)
            A1.append(A_)

        del_idx = list(range(0, int((self.p - 1) / 2))) + list(range(self.n - int((self.p - 1) / 2), self.n))
        A__ = np.delete(A1, np.array(del_idx), axis=0)
        # print(np.shape(A__))
        Y__ = np.delete(Y, np.array(del_idx), axis=0)
        Y1 = np.delete(Y__, np.where(Y__ == 6)[0], axis=0)
        A1 = np.delete(A__, np.where(Y__ == 6)[0], axis=0)
        Y0 = np.delete(Y1, np.where(Y1 == 7)[0], axis=0)
        A = np.delete(A1, np.where(Y1 == 7)[0], axis=0)
        # print(np.shape(A))
        Y_ = encoder.fit_transform(Y0)
        print(len(np.where(Y_ == 0)[0]))
        print(len(np.where(Y_ == 1)[0]))
        print(len(np.where(Y_ == 2)[0]))
        self.A, self.Y_ = np.array(A), np.array(Y_)
        #A, Y_ = np.array(A1), np.array(Y)

        indices = np.arange(np.shape(self.A)[0])
        self.A_train_val, self.A_test, self.Y_train_val, self.Y_test, idx_tv, idx_te = \
            train_test_split(self.A, self.Y_, indices, test_size=0.9, stratify=self.Y_)

        self.A_tv_lab, self.A_tv_un, self.Y_tv_lab, self.Y_tv_un, idx_l, idx_u = \
            train_test_split(self.A_train_val, self.Y_train_val, idx_tv, test_size=1-0.1*self.per, stratify=self.Y_train_val)

        def preprocess_labels(y):
            return y.reshape(-1, 1)

        # tv数据集带标签部分
        self.Y_tv_lab = preprocess_labels(self.Y_tv_lab)
        # tv数据集不带标签部分
        self.Y_tv_un = preprocess_labels(self.Y_tv_un)
        # 测试集
        self.Y_test = preprocess_labels(self.Y_test)

    # Create the labelled samples
    def batch_labeled(self, batch_size, X, Y):
        idx = np.random.randint(0, len(Y), batch_size)
        samples, labels = X[idx], Y[idx]
        return samples, labels, idx

    # Create the unlabelled samples
    def batch_unlabeled(self, batch_size):
        X, Y = self.A_test, self.Y_test
        idx = np.random.randint(0, np.shape(X)[0], batch_size)
        samples, labels = X[idx], Y[idx]
        return samples, labels

    # Create the test set
    def test_set(self):
        return self.A_test, self.Y_test


dataset = Dataset(para_size, num_samples_ori, num_features, batch_size, per)
num_samples = np.shape(dataset.A)[0]
num_classes = len(np.unique(dataset.Y_))


def plt_curves(data, figsize, dpi, sup1):
    N, d = data.shape
    fig, axes = plt.subplots(1, d-1, sharex=False, sharey=True, facecolor='white', figsize=figsize, dpi=dpi)
    colors = ['black', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse',
              'dodgerblue', 'orange', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold']
    title = ['DEPTH', 'CAL', 'SP', 'AC', 'GR', 'COND', 'RLML', 'RNML', 'R25', 'R4']

    for i in range(1, d):
        axes[i-1].plot(data[data.columns[i]], list(range(1, para_size + 1)), c=colors[i-1])
        axes[i-1].set_xlabel(title[i], fontsize=15.5, fontweight=600)
        axes[i-1].set_xlim(-2, 3)
        axes[i-1].set_ylim(N, 1)

    fig.suptitle(sup1, fontsize=23, fontweight=700)


def plt_curves_all(data, figsize, dpi, sup):
    N, d = data.shape
    plt.tight_layout()
    fig, axes = plt.subplots(1, d-1, sharex=False, sharey=True, facecolor='white', figsize=figsize, dpi=dpi)
    colors = ['black', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse',
              'dodgerblue', 'orange', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold']
    title = ['DEPTH', 'CAL', 'SP', 'AC', 'GR', 'COND', 'RLML', 'RNML', 'R25', 'R4']
    dep = data[data.columns[0]]

    for i in range(1, d):
        axes[i-1].plot(data[data.columns[i]], data[data.columns[0]], c=colors[i-1])
        axes[i-1].set_xlabel(title[i], fontsize=13)
        axes[i-1].set_ylim(dep.max(), dep.min())

    # for j in range(0, )
    #     axes[d-1].plot(1, )
    axes[0].set_ylabel('Depth', fontsize=14)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0)
    # fig.suptitle(sup, fontsize=18, fontweight=700)

curves = list(range(num_features))
real_sam_ = pd.DataFrame(SAM_R)
real_sam = real_sam_.iloc[:, curves]
plt_curves_all(real_sam, figsize=(10, 7), dpi=450, sup='a')
plt.savefig('./images' + '/dataset1.jpg')

def bce_loss(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

def cce_loss(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)

def build_generator(z_dim, para_size, num_features):

    model = Sequential()

    # Reshape input into 7x7x256 tensor via a fully connected layer
    model.add(Dense(256 * int(para_size / 5) * int(num_features / 5), input_dim=z_dim,
                    #kernel_initializer=tf.random_normal_initializer(seed=1)
                    kernel_initializer=initializers.he_normal()
                    #kernel_initializer=initializers.RandomNormal(seed=1)
                    ))
    model.add(Reshape((int(para_size / 5), int(num_features / 5), 256)))

    # Transposed convolution layer, from 7x7x256 into 14x14x128 tensor
    model.add(Conv2DTranspose(128, kernel_size=3, strides=5, padding='same',
                              #kernel_initializer=tf.random_normal_initializer(seed=1)
                              kernel_initializer=initializers.he_normal()
                              #kernel_initializer=initializers.RandomNormal(seed=1)
                              ))

    # Batch normalization
    model.add(BatchNormalization())

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=alpha))

    model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding='same',
                              # kernel_initializer=tf.random_normal_initializer(seed=1)
                              kernel_initializer=initializers.he_normal()
                              # kernel_initializer=initializers.RandomNormal(seed=1)
                              ))

    # Batch normalization
    model.add(BatchNormalization())

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=alpha))

    # Transposed convolution layer, from 14x14x64 to 28x28x1 tensor
    model.add(Conv2DTranspose(1, kernel_size=3, strides=1, padding='same',
                              #kernel_initializer=tf.random_normal_initializer(seed=1)
                              kernel_initializer=initializers.he_normal()
                              #kernel_initializer=initializers.RandomNormal(seed=1)
                              ))

    # Output layer with tanh activation
    model.add(Activation('tanh'))

    # model.summary()
    # plot_model(model, to_file='generator_2d.png', show_shapes=True)

    return model


def build_discriminator(sam_shape, dropout):

    model = Sequential()

    # Convolutional layer, from 28x28x1 into 14x14x32 tensor
    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=sam_shape, padding='same',
                     # kernel_initializer=initializers.he_normal()
                     ))

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=alpha))

    # Convolutional layer, from 14x14x32 into 7x7x64 tensor
    model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=sam_shape, padding='same',
                     # kernel_initializer=initializers.he_normal()
                     ))

    # Batch normalization
    model.add(BatchNormalization())

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=alpha))

    # Convolutional layer, from 7x7x64 tensor into 3x3x128 tensor
    model.add(Conv2D(128, kernel_size=3, strides=2, input_shape=sam_shape, padding='same',
                     # kernel_initializer=initializers.he_normal()
                     ))

    # Batch normalization
    model.add(BatchNormalization())

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=alpha))

    # Dropout
    model.add(Dropout(dropout))

    # Flatten the tensor
    model.add(Flatten())

    # Fully connected layer with num_classes neurons
    model.add(Dense(num_classes))

    model.summary()
    # plot_model(model, to_file='discriminator_2d.png', show_shapes=True)

    return model


def build_discriminator_supervised(discriminator_net):

    model = Sequential()

    model.add(discriminator_net)

    # Softmax activation, giving predicted probability distribution over the real classes
    model.add(Activation('softmax'))

    return model


def build_discriminator_unsupervised(discriminator_net):

    model = Sequential()

    model.add(discriminator_net)

    def predict(x):
        # Transform distribution over real classes into a binary real-vs-fake probability
        prediction = 1.0 - (1.0 / (K.sum(K.exp(x), axis=-1, keepdims=True) + 1.0))
        return prediction

    # 'Real-vs-fake' output neuron defined above
    model.add(Lambda(predict))

    # model.summary()

    return model


def build_sgan(generator, discriminator):

    model = Sequential()

    # Combined Generator -> Discriminator model
    model.add(generator)
    model.add(discriminator)

    # model.summary()
    # plot_model(model, to_file='sgan_2d.png', show_shapes=True)

    return model
    

# Set hyperparameters
iterations = 8000
sample_interval = 200

TR, VAL = [], []
for tr_idx, va_idx in SKF.split(dataset.A_tv_lab, dataset.Y_tv_lab):
    TR.append(tr_idx)
    VAL.append(va_idx)

=======
# coding: utf-8

import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
np.random.seed(1)
import tensorflow as tf
from tfdeterminism import patch
patch()
tf.compat.v1.set_random_seed(1)

import random
random.seed(1)
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONHASHSEED"] = "1"

plt.rcParams['font.sans-serif'] = ['Arial']    # 中文
plt.rcParams['axes.unicode_minus'] = False

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

import keras
from keras import backend as K
from keras import initializers
import keras_metrics as km

from keras.layers import (Activation, BatchNormalization, Dense,
                          Dropout, Flatten, Lambda, Reshape)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model

from sklearn import preprocessing, neighbors
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import *
from sklearn.metrics import *
from sklearn.manifold import TSNE
from keras.optimizers import Adam

from scipy.stats import yeojohnson

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
K.set_session(sess)
#

# 定义参数（1%）(OPT1和2反过来)
# [0.9058068  0.49548646 0.76281494]
# 0.7213693999338026
# [0.91220440  0.71639344 0.79258816]
# 0.8240620036555113
batch_size = 16
para_size = 25
channels = 1
rho = 0.8   # 0.5,0.8,0.5
z_dim = 60   # 60,60,60          # 噪声向量维度大小
lr_1 = 0.0002       # 生成器，1e-3, 2e-4, 5e-3
lr_2 = 0.0002        # 判别器, 1e-3, 2e-4, 5e-4
learning_rate = 1e-3
alpha = 0.01
gamma = 1
dropout = 0.5     # 0.7, 0.5
mt_size = 15
beta1 = 0.9
per = 1

# 定义参数（3%）(反过来)
# [0.92230708 0.74695122 0.8045063 ]
# 0.8245881998708651
# [0.93865602 0.83244882 0.84983819 ]
# 0.8736476779534469
# batch_size = 32
# para_size = 25
# channels = 1
# rho = 0.5
# z_dim = 50  # 50          # 噪声向量维度大小
# learning_rate = 1e-3  # 2e-4
# lr_1 = 0.0005        # 生成器 5e-4, 0.015
# lr_2 = 0.0002        # 判别器 2e-4, 0.001
# alpha = 0.01
# gamma = 1
# dropout = 0.5
# mt_size = 15
# beta1 = 0.9
# per = 3

# 定义参数（5%）(211,221,221)(正常顺序)
# [0.93143084 0.8119891  0.79667475]
# 0.8484982286658958
# [0.95581058 0.87175452 0.86848958]
# 0.8986848954698211
batch_size = 64
para_size = 25
channels = 1
rho = 0.5   # 0.35,0.5
z_dim = 60    # 40, 60,70          # 噪声向量维度大小
learning_rate = 5e-4  #2e-4
lr_1 = 0.0005        # 生成器, 3e-4, 5e-4,5e-3
lr_2 = 0.0001       # 判别器, 1e-4, 1e-4, 5e-4
alpha = 0.01
gamma = 1
dropout = 0.5
mt_size = 15
beta1 = 0.9
per = 5

# 定义参数（9%）(OPT1和2反过来)
# [0.96038035 0.92002935 0.88792238]
# 0.9227773592497087
# [0.96373197 0.93026706 0.89458689]
# 0.9295286413346789
# batch_size = 128
# para_size = 25
# channels = 1
# rho = 0.8   # 0.75,0.8
# z_dim = 60  # 50,60          # 噪声向量维度大小
# learning_rate = 2e-4  #2e-4
# lr_1 = 2e-4       # 生成器
# lr_2 = 2e-4       # 判别器
# alpha = 0.01
# gamma = 1
# dropout = 0.5      # 0.5
# mt_size = 15
# beta1 = 0.9
# per = 9

# 定义参数（7%）
# [0.93991194 0.87472202 0.82860147]
# 0.8810784761183038
# [0.95625169 0.91793755 0.87997453]
# 0.9180545878711804
batch_size = 64
para_size = 25
channels = 1
rho = 0.6     # 0.6
z_dim = 85    # 80,85 ,90        # 噪声向量维度大小
learning_rate = 2e-4  #2e-4
lr_1 = 2e-4       # 生成器,8e-4
lr_2 = 2e-4        # 判别器
alpha = 0.01
gamma = 1
dropout = 0.5
mt_size = 15
beta1 = 0.9
per = 7

# bo20
# batch_size = 32
# para_size = 25
# channels = 1
# rho = 0.5    # 0.6
# z_dim = 45    # 80         # 噪声向量维度大小
# learning_rate = 2e-4  #2e-4
# lr_1 = 2e-4       # 生成器,8e-4
# lr_2 = 2e-4        # 判别器
# alpha = 0.01
# gamma = 1
# dropout = 0.3
# mt_size = 15
# beta1 = 0.9
# per = 5

# bo5    0.8631
# batch_size = 64
# para_size = 25
# channels = 1
# rho = 0.8    # 0.6
# z_dim = 50    # 80         # 噪声向量维度大小
# learning_rate = 2e-4  #2e-4
# lr_1 = 4e-4       # 生成器,8e-4
# lr_2 = 2e-4        # 判别器
# alpha = 0.01
# gamma = 1
# dropout = 0.2
# mt_size = 15
# beta1 = 0.9
# per = 5

# bo19
# batch_size = 64
# para_size = 25
# channels = 1
# rho = 0.5  # 0.35,0.5
# z_dim = 60    # 40, 60,70          # 噪声向量维度大小
# learning_rate = 5e-4  #2e-4
# lr_1 = 0.0004       # 生成器, 3e-4, 5e-4,5e-3
# lr_2 = 0.0002       # 判别器, 1e-4, 1e-4, 5e-4
# alpha = 0.01
# gamma = 1
# dropout = 0.5
# mt_size = 15
# beta1 = 0.9
# per = 5

# boxie_24
# batch_size = 128
# para_size = 25
# channels = 1
# rho = 0.8  # 0.35,0.5, 0.8
# z_dim = 60    # 40, 60,70, 60          # 噪声向量维度大小
# learning_rate = 5e-4  #2e-4
# lr_1 = 0.0003     # 生成器, 3e-4, 5e-4,5e-3  3
# lr_2 = 0.00015       # 判别器, 1e-4, 1e-4, 5e-4 ,1.5
# alpha = 0.01
# gamma = 1
# dropout = 0.5   #0.5
# mt_size = 15
# beta1 = 0.9
# per = 5


print(per)

# 遍历文件夹
def scan_file(path):
    d = []
    for root, dirs, files in os.walk(path):
        d.append(root)

    return d[1: len(d)]

filelist_com = scan_file('images/comparison')
del_root = [i for i in filelist_com if len(i) <= len(filelist_com[0])]
for dr in del_root:
    filelist_com.remove(dr)

# 导入数据
path0 = r'3.xlsx'

def loaddata(path):
    data = pd.read_excel(path).values
    Y0 = data[:, -1]
    X0 = data[:, : -1]
    Y0[np.where(Y0 == 4)[0]] = 3
    Y = Y0.copy()
    X_ = X0.copy()
    # X__ = YJ_TRANS(X_)
    mm_tool = StandardScaler()
    X = mm_tool.fit_transform(X_)
    return X0, X, Y, X0[:, 0]

SAM_R, SAM, LITHO, DEP = loaddata(path0)
# dep_1, dep_2 = np.where(DEP == 1036)[0][0], np.where(DEP == 1123)[0][0]
# # X_sel = SAM_R[dep_1: dep_2 + 1]
# # Y_sel = LITHO[dep_1: dep_2 + 1]
#
# X_sel = SAM_R[dep_1 - 1: dep_2 + 1]
# Y_sel = LITHO[dep_1 - 1: dep_2 + 1]

encoder = LabelEncoder()

def get_order(Y):
    Y_sel_1, Y_sel_2 = Y[1: -1], Y[0: -2]
    location1 = np.where(Y_sel_1 != Y_sel_2)[0]
    location = location1[::-1]
    l1 = [len(Y_sel_1) - location[0]]
    l2 = []
    l3 = [location[-1]]
    for x in range(len(location)-1):
        l2.append(location[x] - location[x+1])
    location0 = np.array(l1 + l2 + l3)
    Y_sel_ = encoder.fit_transform(Y_sel_2)
    Y_sel__ = encoder.fit_transform(Y_sel_1)
    Y_sel_2 = Y_sel_.copy()
    Y_sel_1 = Y_sel__.copy()
    return location, location0, Y_sel_1, Y_sel_2

num_samples_ori, num_features = np.shape(SAM)  # 数据集中类别的数量

X = SAM.reshape(len(LITHO), num_features)
B = np.zeros(len(LITHO))
max_X, min_X = [], []
for i in range(num_features):
    B = X[:, i]
    max_X.append(np.max(B))
    min_X.append(np.min(B))

Y = LITHO.copy()

sam_shape = (para_size, num_features, channels)


def getRecall(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))                  #TP
    P=K.sum(K.round(K.clip(y_true, 0, 1)))
    FN = P-TP #FN=P-TP
    recall = TP / (TP + FN + K.epsilon())#TP/(TP+FN)
    return recall

def getPrecision(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))                  #TP
    N = (-1)*K.sum(K.round(K.clip(y_true-K.ones_like(y_true), -1, 0)))   #N
    TN=K.sum(K.round(K.clip((y_true-K.ones_like(y_true))*(y_pred-K.ones_like(y_pred)), 0, 1)))#TN
    FP=N-TN
    precision = TP / (TP + FP + K.epsilon())#TT/P
    return precision

def getf1(y_true, y_pred):
    prec = getPrecision(y_true, y_pred)
    rec = getRecall(y_true, y_pred)
    f1 = 2 * prec * rec / (prec + rec)
    return f1

def plot_embedding_disc(result, y_train, y_test, title, per, name, N=.1, M=.1):  # 传入1083个2维数据，1083个标签，图表标题
    nor_tool = MinMaxScaler(feature_range=[0, 1])
    data = nor_tool.fit_transform(result)
    x_train, x_test = data[: len(y_train), :], data[len(y_train): np.shape(result)[0], :]
    x1_min, x2_min = x_test.min(axis=0)  # 列最小值
    x1_max, x2_max = x_test.max(axis=0)  # 列最大值
    print(x1_min, x2_min, x1_max, x2_max)

    t1, t2 = np.meshgrid(np.arange(x1_min, x1_max, N), np.arange(x2_min, x2_max, M))
    x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
    mesh_input = np.c_[x1.ravel(), x2.ravel()]
    # knn = KNeighborsClassifier(n_neighbors=5)
    # knn.fit(x_train, y_train)
    # y_predict = knn.predict(mesh_input)
    # model_d = Sequential()
    # model_d.add(Dense(num_classes))
    # model_d.add(Activation('softmax'))
    # y_predict = np.argmax(model_d.predict(mesh_input), axis=1)
    # print(model_d.predict(mesh_input))

    plt.figure()  # 创建一个画布
    # cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    markers = ['o', '^', 'D']
    labels = ['RS', 'OS', 'SA']
    colors = ['r', 'g', 'b']
    # plt.pcolormesh(x1, x2, np.array(y_predict).reshape(x1.shape), cmap=cm_light)
    for marker, lab, col in zip(markers, np.unique(y_test), colors):
        plt.scatter(x_test[:, 0].real[y_test == lab],
                    x_test[:, 1].real[y_test == lab],
                    marker=marker, s=15,
                    alpha=0.5, label=labels[int(lab)],
                    edgecolors='k', )
    plt.legend(loc='upper right', borderpad=0.5, fancybox=True)

    plt.title(title, fontdict={'weight': 'heavy', 'size': 14})  # 设置标题
    plt.savefig('./images/tsne/per{}'.format(per)+'/disc'+'/{}.jpg'.format(name))


def plot_embedding_gen(result, label, num, title, per, name, N=100, M=100):  # 传入1083个2维数据，1083个标签，图表标题
    nor_tool = MinMaxScaler(feature_range=[0, 1])
    data = nor_tool.fit_transform(result)

    plt.figure(dpi=500)  # 创建一个画布
    markers = ['o', '^', 's', 'o', '^', 's', 'x']
    labels = ['Real-RS', 'Real-OS', 'Real-SA', 'Real-unlabelled', 'Real-unlabelled', 'Real-unlabelled', 'Fake']
    colors = ['#CD2626', '#228B22', '#4682B4', '#CD2626', '#228B22', '#4682B4', '#6A5ACD']
    edges = ['k', 'k', 'k', None, None, None, None]
    for lab in np.unique(label):
        plt.scatter(data[:, 0].real[label == lab],
                    data[:, 1].real[label == lab],
                    marker=markers[int(lab)], s=25,
                    alpha=0.8, label=labels[int(lab)],
                    color=colors[int(lab)],
                    edgecolors=edges[int(lab)],
                    linewidths=1.5)
    plt.legend(loc='best', prop={'size': 13}, borderpad=0.5, fancybox=True)

    plt.title(title, fontdict={'weight': 600, 'size': 19.5})  # 设置标题
    plt.savefig('./images/tsne/per{}'.format(per)+'/gen/gen_{}'.format(num)+'/{}.jpg'.format(name))


## 权重矩阵
def BUILD_WEIGHT_MAT(Y,  # a label matrix with shape (n,1) or (n,)
                     rho=0.,  # Balance the rare class
                     additional_weights=[],  # additional weights for some samples
                     show_report=True  # Show report if true
                     ):
    # X0: Get basic information
    set_class, num_per_class = np.unique(Y, return_counts=True)  # set of classes and number per class
    num_sample = len(Y)

    if -1 in set_class:
        weight_per_class = 1.0 / (num_per_class ** rho)
        weight_per_class[np.where(set_class == -1)[0]] = 0
    else:
        weight_per_class = 1.0 / (num_per_class ** rho)

    weight_per_class = weight_per_class / sum(weight_per_class)

    if show_report:
        print('# Classes:', set_class, '\n')
        print('# Numbers:', num_per_class, '\n')
        print('# Weights:', weight_per_class, '\n')

    # Build weight_matrix 'C'
    tv = np.zeros(num_sample)  # diagonal elements for 'C'
    for i in set_class:
        tv[list(np.where(Y == i)[0])] = weight_per_class[np.where(set_class == i)[0]]
    if len(additional_weights) > 0:
        tv = tv * additional_weights
    C = tv

    return set_class, C, weight_per_class, num_per_class

SKF = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)       # shuffle:是否在每次分割之前打乱顺序

class Dataset:
    def __init__(self, para_size, num_samples_ori, num_features, batch_size, per):
        self.p = para_size
        self.c = num_features
        self.n = num_samples_ori
        self.b = batch_size
        self.per = per

        A1 = []

        for i in range(self.n):
            if (i >= (int((self.p - 1) / 2))) & (i <= int((self.n - (self.p - 1) / 2) - 1)):
                idx1, idx2 = int((i - (self.p - 1) / 2)), int((i + (self.p - 1) / 2 + 1))
                A0 = X[idx1: idx2, :]
            elif i < (int((self.p - 1) / 2)):
                idx1, idx2 = int(((self.p - 1) / 2) + 1), int((i + (self.p - 1) / 2 + 1))
                A0 = np.vstack((X[int(i + 1): idx1, :], X[0: idx2, :]))
            else:
                idx1, idx2 = int((i - (self.p - 1) / 2)), int(self.n - (self.p - 1) / 2 - 1)
                A0 = np.vstack((X[idx1:, :], X[idx2: int(i), :]))

            A_ = A0.reshape(self.p, self.c, 1)
            A1.append(A_)

        del_idx = list(range(0, int((self.p - 1) / 2))) + list(range(self.n - int((self.p - 1) / 2), self.n))
        A__ = np.delete(A1, np.array(del_idx), axis=0)
        # print(np.shape(A__))
        Y__ = np.delete(Y, np.array(del_idx), axis=0)
        Y1 = np.delete(Y__, np.where(Y__ == 6)[0], axis=0)
        A1 = np.delete(A__, np.where(Y__ == 6)[0], axis=0)
        Y0 = np.delete(Y1, np.where(Y1 == 7)[0], axis=0)
        A = np.delete(A1, np.where(Y1 == 7)[0], axis=0)
        # print(np.shape(A))
        Y_ = encoder.fit_transform(Y0)
        print(len(np.where(Y_ == 0)[0]))
        print(len(np.where(Y_ == 1)[0]))
        print(len(np.where(Y_ == 2)[0]))
        self.A, self.Y_ = np.array(A), np.array(Y_)
        #A, Y_ = np.array(A1), np.array(Y)

        indices = np.arange(np.shape(self.A)[0])
        self.A_train_val, self.A_test, self.Y_train_val, self.Y_test, idx_tv, idx_te = \
            train_test_split(self.A, self.Y_, indices, test_size=0.9, stratify=self.Y_)

        self.A_tv_lab, self.A_tv_un, self.Y_tv_lab, self.Y_tv_un, idx_l, idx_u = \
            train_test_split(self.A_train_val, self.Y_train_val, idx_tv, test_size=1-0.1*self.per, stratify=self.Y_train_val)

        def preprocess_labels(y):
            return y.reshape(-1, 1)  # 将元素转换成一列

        # tv数据集带标签部分
        self.Y_tv_lab = preprocess_labels(self.Y_tv_lab)
        # tv数据集不带标签部分
        self.Y_tv_un = preprocess_labels(self.Y_tv_un)
        # 测试集
        self.Y_test = preprocess_labels(self.Y_test)

    def batch_labeled(self, batch_size, X, Y):
        # 获取随机批量的有标签图像及其标签
        idx = np.random.randint(0, len(Y), batch_size)
        samples, labels = X[idx], Y[idx]
        return samples, labels, idx

    def batch_unlabeled(self, batch_size):
        # 获取随机批量的无标签图像
        X, Y = self.A_test, self.Y_test
        idx = np.random.randint(0, np.shape(X)[0], batch_size)
        samples, labels = X[idx], Y[idx]
        return samples, labels

    def test_set(self):
        return self.A_test, self.Y_test


dataset = Dataset(para_size, num_samples_ori, num_features, batch_size, per)
num_samples = np.shape(dataset.A)[0]
num_classes = len(np.unique(dataset.Y_))
# np.save('boxie_sam.npy', dataset.A)
# np.save('boxie_lab.npy', dataset.Y_)


def plt_curves(data, figsize, dpi, sup1):
    N, d = data.shape
    fig, axes = plt.subplots(1, d-1, sharex=False, sharey=True, facecolor='white', figsize=figsize, dpi=dpi)
    colors = ['black', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse',
              'dodgerblue', 'orange', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold']
    title = ['DEPTH', 'CAL', 'SP', 'AC', 'GR', 'COND', 'RLML', 'RNML', 'R25', 'R4']

    for i in range(1, d):
        axes[i-1].plot(data[data.columns[i]], list(range(1, para_size + 1)), c=colors[i-1])
        axes[i-1].set_xlabel(title[i], fontsize=15.5, fontweight=600)
        axes[i-1].set_xlim(-2, 3)
        axes[i-1].set_ylim(N, 1)

    fig.suptitle(sup1, fontsize=23, fontweight=700)


def plt_curves_all(data, figsize, dpi, sup):
    N, d = data.shape
    plt.tight_layout()
    fig, axes = plt.subplots(1, d-1, sharex=False, sharey=True, facecolor='white', figsize=figsize, dpi=dpi)
    colors = ['black', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse',
              'dodgerblue', 'orange', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold']
    title = ['DEPTH', 'CAL', 'SP', 'AC', 'GR', 'COND', 'RLML', 'RNML', 'R25', 'R4']
    dep = data[data.columns[0]]

    for i in range(1, d):
        axes[i-1].plot(data[data.columns[i]], data[data.columns[0]], c=colors[i-1])
        axes[i-1].set_xlabel(title[i], fontsize=13)
        axes[i-1].set_ylim(dep.max(), dep.min())

    # for j in range(0, )
    #     axes[d-1].plot(1, )
    axes[0].set_ylabel('Depth', fontsize=14)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0)
    # fig.suptitle(sup, fontsize=18, fontweight=700)

curves = list(range(num_features))
real_sam_ = pd.DataFrame(SAM_R)
real_sam = real_sam_.iloc[:, curves]
plt_curves_all(real_sam, figsize=(10, 7), dpi=450, sup='a')
plt.savefig('./images' + '/dataset1.jpg')

# def huber_loss(y_true, y_pred):
#     delta = 1.0
#     resi = K.abs(y_true - y_pred)
#     diff = 0.5 * ((y_true - y_pred) ** 2)
#     loss1 = tf.where(resi < delta, diff, delta * resi - 0.5 * (delta ** 2))
#     loss1_ = K.sum(loss1) * 1e-3
#     loss2 = K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)
#     return loss2

def bce_loss(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

def cce_loss(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)

def build_generator(z_dim, para_size, num_features):

    model = Sequential()

    # Reshape input into 7x7x256 tensor via a fully connected layer
    model.add(Dense(256 * int(para_size / 5) * int(num_features / 5), input_dim=z_dim,
                    #kernel_initializer=tf.random_normal_initializer(seed=1)
                    kernel_initializer=initializers.he_normal()
                    #kernel_initializer=initializers.RandomNormal(seed=1)
                    ))
    model.add(Reshape((int(para_size / 5), int(num_features / 5), 256)))

    # Transposed convolution layer, from 7x7x256 into 14x14x128 tensor
    model.add(Conv2DTranspose(128, kernel_size=3, strides=5, padding='same',
                              #kernel_initializer=tf.random_normal_initializer(seed=1)
                              kernel_initializer=initializers.he_normal()
                              #kernel_initializer=initializers.RandomNormal(seed=1)
                              ))

    # Batch normalization
    model.add(BatchNormalization())

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=alpha))

    model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding='same',
                              # kernel_initializer=tf.random_normal_initializer(seed=1)
                              kernel_initializer=initializers.he_normal()
                              # kernel_initializer=initializers.RandomNormal(seed=1)
                              ))

    # Batch normalization
    model.add(BatchNormalization())

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=alpha))

    # Transposed convolution layer, from 14x14x64 to 28x28x1 tensor
    model.add(Conv2DTranspose(1, kernel_size=3, strides=1, padding='same',
                              #kernel_initializer=tf.random_normal_initializer(seed=1)
                              kernel_initializer=initializers.he_normal()
                              #kernel_initializer=initializers.RandomNormal(seed=1)
                              ))

    # Output layer with tanh activation
    model.add(Activation('tanh'))

    # model.summary()
    # plot_model(model, to_file='generator_2d.png', show_shapes=True)

    return model


def build_discriminator(sam_shape, dropout):

    model = Sequential()

    # Convolutional layer, from 28x28x1 into 14x14x32 tensor
    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=sam_shape, padding='same',
                     # kernel_initializer=initializers.he_normal()
                     ))

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=alpha))

    # Convolutional layer, from 14x14x32 into 7x7x64 tensor
    model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=sam_shape, padding='same',
                     # kernel_initializer=initializers.he_normal()
                     ))

    # Batch normalization
    model.add(BatchNormalization())

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=alpha))

    # Convolutional layer, from 7x7x64 tensor into 3x3x128 tensor
    model.add(Conv2D(128, kernel_size=3, strides=2, input_shape=sam_shape, padding='same',
                     # kernel_initializer=initializers.he_normal()
                     ))

    # Batch normalization
    model.add(BatchNormalization())

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=alpha))

    # Dropout
    model.add(Dropout(dropout))

    # Flatten the tensor
    model.add(Flatten())

    # Fully connected layer with num_classes neurons
    model.add(Dense(num_classes))

    model.summary()
    # plot_model(model, to_file='discriminator_2d.png', show_shapes=True)

    return model


def build_discriminator_supervised(discriminator_net):

    model = Sequential()

    model.add(discriminator_net)

    # Softmax activation, giving predicted probability distribution over the real classes
    model.add(Activation('softmax'))

    return model


def build_discriminator_unsupervised(discriminator_net):

    model = Sequential()

    model.add(discriminator_net)

    def predict(x):
        # Transform distribution over real classes into a binary real-vs-fake probability
        prediction = 1.0 - (1.0 / (K.sum(K.exp(x), axis=-1, keepdims=True) + 1.0))
        return prediction

    # 'Real-vs-fake' output neuron defined above
    model.add(Lambda(predict))

    # model.summary()

    return model


def build_sgan(generator, discriminator):

    model = Sequential()

    # Combined Generator -> Discriminator model
    model.add(generator)
    model.add(discriminator)

    # model.summary()
    # plot_model(model, to_file='sgan_2d.png', show_shapes=True)

    return model
    

# Set hyperparameters
iterations = 8000
sample_interval = 200

TR, VAL = [], []
for tr_idx, va_idx in SKF.split(dataset.A_tv_lab, dataset.Y_tv_lab):
    TR.append(tr_idx)
    VAL.append(va_idx)

>>>>>>> 3419a1cb4c245314f14c64d4238cf2b661df991b
TR, VAL = np.array(TR), np.array(VAL)