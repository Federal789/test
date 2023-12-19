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

style.use('seaborn')

fig = plt.figure(dpi=500)

# F1-score随标签比例变化的趋势图
recall1 = np.array([0.8536, 0.8923, 0.9094, 0.9203])
xlabel = [2.5, 5, 10, 20]

x = list(range(0, len(recall1)))
plt.plot(x, recall1, color='darkred', alpha=0.5, linestyle='-', linewidth=3, marker='*',
         markeredgecolor='gold', markersize='13', markeredgewidth=5, label='TEA-SDA')
plt.legend(loc='upper left', borderpad=0.8, fancybox=True, fontsize=12)
# plt.bar(0, first, width=1.5)
plt.grid(alpha=0.5, linestyle='-.', linewidth=1.5, axis='both')
ax2 = plt.subplot(111)
ax2.set_ylim([0.85, 0.925])
ax2.set_xlim([-0.2, len(x)-0.5])
plt.xticks(range(len(xlabel)), labels=xlabel, fontsize=12, fontweight=550)
plt.yticks(fontsize=12, fontweight=550)
for a, b in zip(x, recall1):
    plt.text(a+0.03, b+0.005, '%.2f' % b, ha='center', va='top', fontdict={'size': 12, 'weight': 600})

def aax(x):
    x0 = (x + 0.2) / (3.5 + 0.2)
    return x0

def aay(y):
    y0 = (y-0.85) / (0.925-0.85)
    return y0

right_bias = 0.22

plt.text(0.2 + 1 * 0 + right_bias, recall1[0]-0.003, '×2', fontdict={'size': 11, 'weight': 550})
plt.text(0.2 + 1 * 1 + right_bias, recall1[1]-0.003, '×2', fontdict={'size': 11, 'weight': 550})
plt.text(0.2 + 1 * 2 + right_bias, recall1[2]-0.003, '×2', fontdict={'size': 11, 'weight': 550})
# plt.text(0.2 + 1 * 3 + right_bias * 2.2, recall1[3]-0.004, '+2%', fontdict={'size': 11, 'weight': 550})
plt.text(1 * 1 + 0.009, 0.5*(recall1[0]+recall1[1]), 'Δ={}'.format(round(recall1[1]-recall1[0], 4)), fontdict={'size': 11, 'weight': 550})
plt.text(1 * 2 + 0.009, 0.5*(recall1[1]+recall1[2]), 'Δ={}'.format(round(recall1[2]-recall1[1], 4)), fontdict={'size': 11, 'weight': 550})
plt.text(1 * 3 + 0.009, 0.5*(recall1[2]+recall1[3]), 'Δ={}'.format(round(recall1[3]-recall1[2], 4)), fontdict={'size': 11, 'weight': 550})


plt.axhline(y=recall1[0], xmin=aax(0), xmax=aax(1), ls='--', lw=2, c='k')
plt.axhline(y=recall1[1], xmin=aax(1), xmax=aax(2), ls='--', lw=2, c='k')
plt.axhline(y=recall1[2], xmin=aax(2), xmax=aax(3), ls='--', lw=2, c='k')
plt.axvline(x=1, ymin=aay(recall1[0]), ymax=aay(recall1[1]), ls='--', lw=2, c='k')
plt.axvline(x=2, ymin=aay(recall1[1]), ymax=aay(recall1[2]), ls='--', lw=2, c='k')
plt.axvline(x=3, ymin=aay(recall1[2]), ymax=aay(recall1[3]), ls='--', lw=2, c='k')

k1 = round((recall1[1]-recall1[0])*100/2, 3)
k2 = round((recall1[2]-recall1[1])*100/2, 3)
k3 = round((recall1[3]-recall1[2])*100/2, 3)

plt.text(0.2 + 1 * 0 + right_bias, recall1[0]+0.022, 'k1', fontdict={'size': 12, 'weight': 550})
plt.text(0.2 + 1 * 1 + right_bias, recall1[1]+0.011, 'k2', fontdict={'size': 12, 'weight': 550})
plt.text(0.2 + 1 * 2 + right_bias, recall1[2]+0.01, 'k3', fontdict={'size': 12, 'weight': 550})

plt.annotate('\n k1={},\n\n k2={}, \n\n k3={} \n'.format(k1, k2, k3),
             xy=(2.3, 0.86), xytext=(2.6, 0.86), weight=550, fontsize=12,
             bbox=dict(boxstyle='round, pad=0.5', fc='cyan', ec='k', lw=1, alpha=0.4))

plt.xlabel('标签比例（%）', fontdict={'weight': 550, 'size': 15})
plt.ylabel('平均F1-score', fontdict={'weight': 550, 'size': 15})

# 保存
plt.tight_layout()
fig.savefig("trendency.png", bbox_inches='tight', pad_inches=0)
plt.close(fig)