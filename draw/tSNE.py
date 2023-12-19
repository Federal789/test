import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import seaborn as sns

sam_A, lab_A = np.load('bo20_sam.npy'), np.load('bo20_lab.npy')
sam_B, lab_B = np.load('bo5_sam.npy'), np.load('bo5_lab.npy')
sam_C, lab_C = np.load('bo19_sam.npy'), np.load('bo19_lab.npy')
sam_D, lab_D = np.load('boxie_sam.npy'), np.load('boxie_lab.npy')
sam_E, lab_E = np.load('bo3_sam.npy'), np.load('bo3_lab.npy')

sam_all_ = np.vstack([sam_A, sam_B, sam_C, sam_D, sam_E])
lab_all = np.hstack([lab_A, lab_B, lab_C, lab_D, lab_E])
sam_all_ = sam_all_.reshape(sam_all_.shape[0], -1)

tsne = TSNE(n_components=2, init='pca', random_state=0)
sam_all = tsne.fit_transform(sam_all_)                     # 进行降维
mm_tool = MinMaxScaler(feature_range=[0, 1])
sam_norm = mm_tool.fit_transform(sam_all)
df_sam = pd.DataFrame(sam_norm, columns=['x', 'y'])

sns.jointplot(x='x', y='y', data=df_sam[0: len(lab_A)], kind='kde', color='r')
sns.jointplot(x='x', y='y', data=df_sam[len(lab_A): len(lab_A) + len(lab_B)], kind='kde', color='green')
sns.jointplot(x='x', y='y', data=df_sam[len(lab_A) + len(lab_B): len(lab_A) + len(lab_B) + len(lab_C)], kind='kde', color='orange')
sns.jointplot(x='x', y='y', data=df_sam[len(lab_A) + len(lab_B) + len(lab_C): len(lab_A) + len(lab_B) + len(lab_C) + len(lab_D)], kind='kde')
sns.jointplot(x='x', y='y', data=df_sam[len(lab_A) + len(lab_B) + len(lab_C) + len(lab_D): ], kind='kde', color='pink')