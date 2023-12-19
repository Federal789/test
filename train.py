import numpy as np

from KF1 import *
from KF2 import *
from KF3 import *

np.random.seed(1)
tf.compat.v1.set_random_seed(1)
random.seed(1)

# import seaborn as sns
# sns.set(font_scale=1.4) # y label 横向

len_param = len(dis_wb1)

new_param = []

for i in range(len_param):
    new_pa = (dis_wb1[i] * (fscore1 ** 1) + dis_wb2[i] * (fscore2 ** 1) + dis_wb3[i] * (fscore3 ** 1))\
            / (1 * (fscore1 ** 1) + 1 * (fscore2 ** 1) + 1 * (fscore3 ** 1))
    #new_pa = (dis_wb1[i] * (fscore1 ** 1) + dis_wb2[i] * (fscore2 ** 1))\
    #        / (1 * (fscore1 ** 1) + 1 * (fscore2 ** 1))
    # new_pa = np.mean(np.array([dis_wb1[i], dis_wb2[i], dis_wb3[i]]), axis=0)
    new_param.append(new_pa)

net_new = build_discriminator(sam_shape, dropout)
NEW_classifier = build_discriminator_supervised(net_new)
NEW_classifier.set_weights(new_param)

x3, y3 = np.array(dataset.A_test), np.array(dataset.Y_test)
y3_pred = NEW_classifier.predict_classes(x3)
y3_pred = y3_pred.reshape(-1, 1)
result_test = classification_report(y3, y3_pred)
print(result_test)
f1_test = f1_score(y3, y3_pred, average=None)
# f1 = np.vstack((f1_test, f1_score(y3, y3_pred, average='macro')))
print(f1_test)
print(np.mean(f1_test))
print(f1_score(y3, y3_pred, average='macro'))

np.save('./images/conf_mat/y_true_{}.npy'.format(per), y3)
np.save('./images/conf_mat/y_pred_{}.npy'.format(per), y3_pred)

plt.rcParams['savefig.dpi'] = 450
conf_mat = confusion_matrix(y3, y3_pred, sample_weight=None)
conf_mat_norm = np.around(conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis], decimals=2)
thresh = conf_mat_norm.max() / 2
plt.matshow(conf_mat_norm, cmap=plt.get_cmap('Blues'))
plt.title('{}% labels'.format(per), ha='center', va='bottom', fontdict={'size': 16, 'weight': 600})
plt.colorbar()
for i in range(len(conf_mat_norm)):
    for j in range(len(conf_mat_norm)):
        plt.annotate(conf_mat_norm[j, i], xy=(i, j), ha='center', va='center', alpha=1, weight=550,
                     size=12, color="white" if conf_mat_norm[j, i] > thresh else "black")
plt.xlabel('Predicted labels', fontdict={'size': 14, 'weight': 500})
plt.ylabel('True labels', fontdict={'size': 14, 'weight': 500})
x = range(0, 3, 1)
plt.xticks(x, ('RS', 'OS', 'SA'), fontsize=14)
plt.yticks(x, ('RS', 'OS', 'SA'), fontsize=14)
plt.tight_layout()
plt.savefig('./images/conf_mat'+'/per{}.jpg'.format(per))
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa


# Save t-SNE
dis_1 = Model(inputs=net_new.inputs,
              outputs=net_new.get_layer('leaky_re_lu_16').output)

dis_2 = Model(inputs=net_new.inputs,
              outputs=net_new.get_layer('leaky_re_lu_17').output)

dis_3 = Model(inputs=net_new.inputs,
              outputs=net_new.get_layer('leaky_re_lu_18').output)

dis_seq = Model(inputs=net_new.inputs,
                outputs=net_new.get_layer('dense_7').output)

dis_1_ = dis_1.predict(x3)
dis_2_ = dis_2.predict(x3)
dis_3_ = dis_3.predict(x3)
dis_1_real = dis_1_.reshape(dis_1_.shape[0], -1)
dis_2_real = dis_2_.reshape(dis_2_.shape[0], -1)
dis_3_real = dis_3_.reshape(dis_3_.shape[0], -1)
dis_seq_real = dis_seq.predict(x3)
dis_sig_real = NEW_classifier.predict(x3)

real_lab = y_test.reshape(len(y_test))
pred_lab = y3_pred.reshape(len(y3_pred))
real_sam = x_test.reshape(len(y_test), -1)

# training samples
x_train, y_train = dataset.A_tv_lab, dataset.Y_tv_lab
dis_1_tr_ = dis_1.predict(x_train)
dis_2_tr_ = dis_2.predict(x_train)
dis_3_tr_ = dis_3.predict(x_train)
dis_1_tr_real = dis_1_tr_.reshape(dis_1_tr_.shape[0], -1)
dis_2_tr_real = dis_2_tr_.reshape(dis_2_tr_.shape[0], -1)
dis_3_tr_real = dis_3_tr_.reshape(dis_3_tr_.shape[0], -1)
dis_seq_tr = dis_seq.predict(x_train)
dis_sig_tr = NEW_classifier.predict(x_train)
tr_lab = y_train.reshape(len(y_train))
tr_sam = x_train.reshape(len(y_train), -1)

# 原始数据
np.random.seed(1)
tsne = TSNE(n_components=2, init='pca', random_state=1)  # n_components将64维降到该维度，默认2
result_ori = tsne.fit_transform(np.vstack((tr_sam, real_sam)))
plot_embedding_disc(result_ori, tr_lab, real_lab, '{}%, t-SNE embedding of original samples'.format(per), per, 'origin')  # 显示数据


# 第一层
np.random.seed(1)
tsne = TSNE(n_components=2, init='pca', random_state=1)  # n_components将64维降到该维度，默认2
result_1 = tsne.fit_transform(np.vstack((dis_1_tr_real, dis_1_real)))
plot_embedding_disc(result_1, tr_lab, real_lab, '{}%, t-SNE embedding of convolutional layer I'.format(per), per, 'conv1')  # 显示数据

# 第二层
np.random.seed(1)
tsne = TSNE(n_components=2, init='pca', random_state=1)  # n_components将64维降到该维度，默认2
result_2 = tsne.fit_transform(np.vstack((dis_2_tr_real, dis_2_real)))
plot_embedding_disc(result_2, tr_lab, real_lab, '{}%, t-SNE embedding of convolutional layer II'.format(per), per, 'conv2')  # 显示数据

# 第三层
np.random.seed(1)
tsne = TSNE(n_components=2, init='pca', random_state=1)  # n_components将64维降到该维度，默认2
result_3 = tsne.fit_transform(np.vstack((dis_3_tr_real, dis_3_real)))
plot_embedding_disc(result_3, tr_lab, real_lab, '{}%, t-SNE embedding of convolutional layer III'.format(per), per, 'conv3')  # 显示数据

# Dense层
np.random.seed(1)
tsne = TSNE(n_components=2, init='pca', random_state=1)  # n_components将64维降到该维度，默认2
result_den = tsne.fit_transform(np.vstack((dis_seq_tr, dis_seq_real)))
plot_embedding_disc(result_den, tr_lab, real_lab, '{}%, t-SNE embedding of the dense layer'.format(per), per, 'dense')  # 显示数据

# 预测结果
np.random.seed(1)
tsne = TSNE(n_components=2, init='pca', random_state=1)  # n_components将64维降到该维度，默认2
result_out = tsne.fit_transform(np.vstack((dis_sig_tr, dis_sig_real)))
plot_embedding_disc(result_out, tr_lab, real_lab, '{}%, t-SNE embedding of the output'.format(per), per, 'output')  # 显示数据