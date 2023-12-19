# coding: utf-8

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


# 雷达图
fig = plt.figure(figsize=(5, 5), dpi=450)

feature = ['RS', 'OS', 'SA', 'Average']
angles = np.linspace(0.3 * np.pi, 2.3 * np.pi, len(feature), endpoint=False)
angles = np.concatenate((angles, [angles[0]]))

value1 = [0.88523219, 0.60422961, 0.78437702, 0.75794627]
value2 = [0.91593813, 0.72698662, 0.80298322, 0.81530266]

value1 = np.concatenate((value1, [value1[0]]))
value2 = np.concatenate((value2, [value2[0]]))

value3 = [0.9243, 0.7855, 0.8371, 0.8489]
value4 = [0.94015957, 0.84367988, 0.84969025,	0.87784323]
value3 = np.concatenate((value3, [value3[0]]))
value4 = np.concatenate((value4, [value4[0]]))

value5 = [0.9417, 0.8536, 0.8499, 0.8818]
value6 = [0.95270988, 0.89427313, 0.87211602, 0.90636634]
value5 = np.concatenate((value5, [value5[0]]))
value6 = np.concatenate((value6, [value6[0]]))

value7 = [0.9445,	0.8869, 0.8655,	0.8989]
value8 = [0.96058981,	0.9263623, 0.88280747,	0.92325319]
value7 = np.concatenate((value7, [value7[0]]))
value8 = np.concatenate((value8, [value8[0]]))

value9 = [0.9496,	0.9225, 0.8630,	0.9117]
value10 = [0.9644765, 0.9393718, 0.89272905, 0.93219245]
value9 = np.concatenate((value9, [value9[0]]))
value10 = np.concatenate((value10, [value10[0]]))
feature = np.concatenate((feature, [feature[0]]))

y_major_locator = MultipleLocator(0.1)

ax1 = fig.add_subplot(111, polar=True)
ax1.plot(angles, value2, 'D-.', linewidth=0.8, markersize=2, label='WA-SSGAN')
ax1.fill(angles, value2, alpha=0.3)

ax1.plot(angles, value1, 'o-', linewidth=0.8, markersize=2, label='Discriminator')
ax1.fill(angles, value1, alpha=0.3)

ax1.set_thetagrids(angles * 180/np.pi, feature, fontsize=16, fontweight=550)

ax1.set_ylim(0.5, 1)
ax1.yaxis.set_major_locator(y_major_locator)
ax1.tick_params('y', labelsize=13)
# ax1.set_title('(a)1%', fontdict={'weight': 600, 'size': 20})
ax1.grid(True)
plt.ylim(0.5, 1)
plt.tight_layout()
fig.savefig("ldt_1.png", bbox_inches='tight', pad_inches=0)
plt.close(fig)


fig = plt.figure(figsize=(5, 5), dpi=450)
ax2 = fig.add_subplot(111, polar=True)
ax2.plot(angles, value4, 'D-.', linewidth=0.8, markersize=2, label='WA-SSGAN')
ax2.fill(angles, value4, alpha=0.3)

ax2.plot(angles, value3, 'o-', linewidth=0.8, markersize=2, label='Discriminator')
ax2.fill(angles, value3, alpha=0.3)

ax2.set_thetagrids(angles * 180/np.pi, feature, fontsize=16, fontweight=550)

ax2.set_ylim(0.5, 1)
ax2.yaxis.set_major_locator(y_major_locator)
ax2.tick_params('y', labelsize=13)
# ax2.set_title('(b)3%', fontdict={'weight': 600, 'size': 20})
ax2.grid(True)
plt.ylim(0.5, 1)
plt.tight_layout()
fig.savefig("ldt_2.png", bbox_inches='tight', pad_inches=0)
plt.close(fig)


fig = plt.figure(figsize=(5, 5), dpi=450)
ax3 = fig.add_subplot(111, polar=True)
ax3.plot(angles, value6, 'D-.', linewidth=0.8, markersize=2, label='WA-SSGAN')
ax3.fill(angles, value6, alpha=0.3)

ax3.plot(angles, value5, 'o-', linewidth=0.8, markersize=2, label='Discriminator')
ax3.fill(angles, value5, alpha=0.3)

ax3.set_thetagrids(angles * 180/np.pi, feature, fontsize=16, fontweight=550)

ax3.set_ylim(0.5, 1)
ax3.yaxis.set_major_locator(y_major_locator)
ax3.tick_params('y', labelsize=13)
# ax3.set_title('(c)5%', fontdict={'weight': 600, 'size': 20})
ax3.grid(True)
plt.ylim(0.5, 1)
plt.tight_layout()
fig.savefig("ldt_3.png", bbox_inches='tight', pad_inches=0)
plt.close(fig)


fig = plt.figure(figsize=(5, 5), dpi=450)
ax4 = fig.add_subplot(111, polar=True)
ax4.plot(angles, value8, 'D-.', linewidth=0.8, markersize=2, label='WA-SSGAN')
ax4.fill(angles, value8, alpha=0.3)

ax4.plot(angles, value7, 'o-', linewidth=0.8, markersize=2, label='Discriminator')
ax4.fill(angles, value7, alpha=0.3)

ax4.set_thetagrids(angles * 180/np.pi, feature, fontsize=16, fontweight=550)

ax4.set_ylim(0.5, 1)
ax4.yaxis.set_major_locator(y_major_locator)
ax4.tick_params('y', labelsize=13)
# ax4.set_title('(d)7%', fontdict={'weight': 600, 'size': 20})
ax4.grid(True)
plt.ylim(0.5, 1)
plt.tight_layout()
fig.savefig("ldt_4.png", bbox_inches='tight', pad_inches=0)
plt.close(fig)


fig = plt.figure(figsize=(5, 5), dpi=450)
ax5 = fig.add_subplot(111, polar=True)
ax5.plot(angles, value10, 'D-.', linewidth=0.8, markersize=2, label='WA-SSGAN')
ax5.fill(angles, value10, alpha=0.3)

ax5.plot(angles, value9, 'o-', linewidth=0.8, markersize=2, label='Discriminator')
ax5.fill(angles, value9, alpha=0.3)

ax5.set_thetagrids(angles * 180/np.pi, feature, fontsize=16, fontweight=550)

ax5.set_ylim(0.5, 1)
ax5.yaxis.set_major_locator(y_major_locator)
ax5.tick_params('y', labelsize=13)
# ax5.set_title('(e)9%', fontdict={'weight': 600, 'size': 20})
ax5.grid(True)
plt.ylim(0.5, 1)
plt.tight_layout()
fig.savefig("ldt_5.png", bbox_inches='tight', pad_inches=0)
plt.close(fig)


fig = plt.figure(figsize=(5, 5), dpi=450)
ax6 = fig.add_subplot(111, polar=False)
ax6.set_frame_on(False)
ax6.get_xaxis().set_visible(False)
ax6.get_yaxis().set_visible(False)
ax6.plot(0, 0, 'D-', linewidth=0.8, markersize=2,  label='WA-SSGAN')
ax6.plot(0, 0, 'o-', linewidth=0.8, markersize=2, label='CNN')
ax6.legend(loc='center', borderpad=5, fancybox=True, prop=dict(size=27))
plt.ylim(0.5, 1)
plt.tight_layout()
fig.savefig("ldt_6.png", bbox_inches='tight', pad_inches=0)
plt.close(fig)

# 保存
# plt.ylim(0.5, 1)
# plt.tight_layout()
# fig.savefig("ldt_all.png", bbox_inches='tight', pad_inches=0)
# plt.close(fig)