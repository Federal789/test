import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as mfm
import seaborn as sns
from matplotlib import style
import matplotlib as mpl
import pygal
import mpl_toolkits.mplot3d
from matplotlib.pyplot import MultipleLocator


plt.rcParams['font.sans-serif'] = ['Arial']    # 中文
plt.rcParams['axes.unicode_minus'] = False
# style.use('Solarize_Light2')

# 多种算法对比
fig = plt.figure(figsize=(10, 7), dpi=500)
labels = ['1', '3', '5', '7', '9']
bar_width = 0.25
leg_prop = {'weight': 600, 'size': 14}


plt.subplot(211)
SVM_L = np.array([0.726,0.785,0.798,0.821, 0.829])
DT_L = np.array([0.699,0.793,0.838,0.866, 0.888])
KNN_L = np.array([0.705,0.774,0.809,0.839, 0.860])
FCN = np.array([0.737,0.817,0.839,0.853, 0.887])
Ours = np.array([0.741,0.830,0.881,0.896, 0.910])


labels = ['1', '3', '5', '7', '9']
bar_width = 0.25

x = np.arange(0, bar_width * (1 + len(labels)) * len(labels), bar_width * (1 + len(labels)))
plt.bar(x + 1 * bar_width, SVM_L, color='white', edgecolor='firebrick', width=bar_width, hatch='++', label='SVM')
plt.bar(x + 2 * bar_width, KNN_L, color='white', edgecolor='SandyBrown', width=bar_width, hatch='O', label='KNN')
plt.bar(x + 3 * bar_width, DT_L, color='white', edgecolor='#339966', width=bar_width, hatch='xx', label='DT')
plt.bar(x + 4 * bar_width, FCN, color='white', edgecolor='cornflowerblue', width=bar_width, hatch='--', label='FCN')
plt.bar(x + 5 * bar_width, Ours, color='white', edgecolor='Navy', width=bar_width, hatch='*', label='CNN')

leg_prop = {'weight': 600, 'size': 14}

sns.despine(left=False, bottom=False)
plt.xticks(x + 3 * bar_width, labels=labels, fontdict={'weight': 550, 'size': 15, 'color': 'k'})
plt.yticks(fontsize=15, fontweight=550, color='k')
plt.legend(prop=leg_prop, loc='lower right', borderpad=0.8, fancybox=True)
plt.xlabel('Label ratio(%)', fontdict={'weight': 550, 'size': 17, 'color': 'k'})
plt.ylabel('Average F1-score', fontdict={'weight': 550, 'size': 17, 'color': 'k'})
plt.title('(a)Comparison of different supervised algorithms', fontdict={'weight': 700, 'size': 20})

index = np.arange(0, bar_width * (1 + len(labels)) * len(labels), bar_width * (1+len(labels)))
index_0 = index + bar_width
index_1 = index_0 + bar_width
index_2 = index_1 + bar_width
index_3 = index_2 + bar_width
index_4 = index_3 + bar_width

index_all = np.transpose(np.vstack((SVM_L, KNN_L, DT_L, FCN, Ours)))
print(np.shape(index_all))

per1 = index_all[:, 0]
per3 = index_all[:, 1]
per5 = index_all[:, 2]
per7 = index_all[:, 3]
per9 = index_all[:, -1]

for index_0, per1 in zip(index_0, per1):
    print(index_0, per1)
    plt.text(index_0, per1, '%.2f' %per1, ha='center', va='bottom', fontsize=9.5)
for index_1, per3 in zip(index_1, per3):
    print(index_1, per3)
    plt.text(index_1, per3, '%.2f' %per3, ha='center', va='bottom', fontsize=9.5)
for index_2, per5 in zip(index_2, per5):
    print(index_2, per5)
    plt.text(index_2, per5, '%.2f' %per5, ha='center', va='bottom', fontsize=9.5)
for index_3, per7 in zip(index_3, per7):
    print(index_3, per7)
    plt.text(index_3, per7, '%.2f' %per7, ha='center', va='bottom', fontsize=9.5)
for index_4, per9 in zip(index_4, per9):
    print(index_4, per9)
    plt.text(index_4, per9, '%.2f' %per9, ha='center', va='bottom', fontsize=9.5)


plt.subplot(212)
'''

'''
# style.use('Solarize_Light2')
fig = plt.figure(figsize=(11, 6), dpi=450)
labels = ['1', '3', '5', '7', '9']
bar_width = 1 / 4
leg_prop = {'weight': 600, 'size': 14}
SVM_L = np.array([0.7470, 0.8271, 0.8416, 0.8581, 0.8607])
DT_L = np.array([0.7255, 0.8216, 0.8344, 0.8693, 0.8830])
KNN_L = np.array([0.7358, 0.8129, 0.8385, 0.8870, 0.8951])
FCN_L = np.array([0.7627, 0.8334, 0.8586, 0.8882, 0.9107])
LN = np.array([0.7865, 0.8208, 0.8314, 0.8603, 0.8648])
MT = np.array([0.7718, 0.8194, 0.8382, 0.8636, 0.8896])
MIX = np.array([0.7986, 0.8536, 0.8815, 0.8984, 0.9138])
Ours = np.array([0.8153, 0.8778, 0.9064, 0.9233, 0.9322])

x = np.arange(0, bar_width * (1 + 8) * len(labels), bar_width * (1 + 8))
plt.bar(x + 1 * bar_width, SVM_L, color='r', edgecolor='k', width=bar_width, label='LapSVM')
plt.bar(x + 2 * bar_width, KNN_L, color='gold', edgecolor='k', width=bar_width, label='LPA+KNN')
plt.bar(x + 3 * bar_width, DT_L, color='#228B22', edgecolor='k', width=bar_width, label='LPA+DT')
plt.bar(x + 4 * bar_width, FCN_L, color='#1E90FF', edgecolor='k', width=bar_width, label='LPA+FCN')
plt.bar(x + 5 * bar_width, LN, color='#7B68EE', edgecolor='k', width=bar_width, label='LadderNet')
plt.bar(x + 6 * bar_width, MT, color='violet', edgecolor='k', width=bar_width, label='Mean Teacher')
plt.bar(x + 7 * bar_width, MIX, color='tan', edgecolor='k', width=bar_width, label='MixMatch')
plt.bar(x + 8 * bar_width, Ours, color='grey', edgecolor='k', width=bar_width, label='WA-SSGAN')

sns.despine(left=False, bottom=False)
plt.xticks(x + 4 * bar_width, labels=labels, fontdict={'weight': 600, 'size': 17, 'color': 'k'})
plt.yticks(fontsize=17, fontweight=600, color='k')
plt.legend(prop=leg_prop, loc='lower right', borderpad=0.8, fancybox=True)
plt.xlabel('Label ratio(%)', fontdict={'weight': 600, 'size': 19.5, 'color': 'k'})
plt.ylabel('Average F1-score', fontdict={'weight': 550, 'size': 17, 'color': 'k'})
# plt.title('Comparison of different semi-supervised algorithms', fontdict={'weight': 700, 'size': 20})

index = np.arange(0, bar_width * (1 + 8) * 8, bar_width * (1 + 8))
index_0 = index + bar_width
index_1 = index_0 + bar_width
index_2 = index_1 + bar_width
index_3 = index_2 + bar_width
index_4 = index_3 + bar_width
index_5 = index_4 + bar_width
index_6 = index_5 + bar_width
index_7 = index_6 + bar_width

index_all = np.transpose(np.vstack((SVM_L, KNN_L, DT_L, FCN_L, LN, MT, MIX, Ours)))

SVM = index_all[:, 0]
KNN = index_all[:, 1]
DT = index_all[:, 2]
FCN = index_all[:, 3]
LAD = index_all[:, 4]
MEAN = index_all[:, 5]
MIX_ = index_all[:, 6]
OURS = index_all[:, 7]

for index_0, s in zip(index_0, SVM):
    plt.text(index_0, s, '%.2f' %s, ha='center', va='bottom', fontsize=8)
for index_1, k in zip(index_1, KNN):
    plt.text(index_1, k, '%.2f' %k, ha='center', va='bottom', fontsize=8)
for index_2, d in zip(index_2, DT):
    plt.text(index_2, d, '%.2f' %d, ha='center', va='bottom', fontsize=8)
for index_3, f in zip(index_3, FCN):
    plt.text(index_3, f, '%.2f' %f, ha='center', va='bottom', fontsize=8)
for index_4, l in zip(index_4, LAD):
    plt.text(index_4, l, '%.2f' %l, ha='center', va='bottom', fontsize=8)
for index_5, m in zip(index_5, MEAN):
    plt.text(index_5, m, '%.2f' %m, ha='center', va='bottom', fontsize=8)
for index_6, M in zip(index_6, MIX_):
    plt.text(index_6, M, '%.2f' %M, ha='center', va='bottom', fontsize=8)
for index_7, o in zip(index_7, OURS):
    plt.text(index_7, o, '%.2f' %o, ha='center', va='bottom', fontsize=8)


plt.tight_layout()
fig.savefig("two_compare.png", bbox_inches='tight', pad_inches=0)
plt.close(fig)