# 超参数变化图

import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 有时没这句会报错


COLOR = ["orangered", "khaki", "mediumspringgreen", "cornflowerblue", "m"]

plt.rcParams['font.sans-serif'] = ['SimHei']    # 中文
plt.rcParams['axes.unicode_minus'] = False

delta = ['0.1', '0.3', '0.5', '1']
lamda = ['8e6', '8e5', '8e4']

# x, y: position
x = list(range(len(lamda)))
y = list(range(len(delta)))

# z = np.array([[0.77056063, 0.78965285, 0.79490085, 0.79644042, 0.79692826],
#  [0.79081967, 0.79888292, 0.81258367, 0.80566406, 0.80014713],
#  [0.7947204, 0.80024501, 0.79086187, 0.78476995, 0.77119695],
#  [0.79501455, 0.80836788, 0.79472139, 0.78701703, 0.77393306],
#  [0.80614144, 0.80504512, 0.78821988, 0.77480796, 0.76356411]])




# z1 = np.array([[0.91593813, 0.72698662, 0.80298322],
#               [0.94015957, 0.84367988, 0.84969025],
#               [0.95270988, 0.89427313, 0.87211602],
#               [0.96058981, 0.9263623, 0.88280747],
#               [0.9644765, 0.9393718, 0.89272905]])

z1 = np.array([[0.8067, 0.8231, 0.8496],
              [0.8169, 0.8385, 0.8687],
              [0.8427, 0.8675, 0.8923],
              [0.8204, 0.8459, 0.8712]
              ])

z1 = np.transpose(z1)

z0 = z1 - 0.80
print(z0)

xx, yy = np.meshgrid(x, y)

color_list = []
for i in range(len(y)):
    c = COLOR[i]
    color_list.append([c] * len(x))
color_list = np.asarray(color_list)

xx_flat, yy_flat, z_flat, color_flat = \
    xx.ravel(), yy.ravel(), z0.T.ravel(), color_list.ravel()

fig = plt.figure(dpi=450)
ax = fig.add_subplot(111, projection="3d")

ax.zaxis.set_rotate_label(False)     #一定要先关掉默认的旋转设置
ax.set_zlabel('Weight (kg)', rotation=90)
ax.set_xlabel('Weight (kg)', rotation=-16)

bar_width = 0.5
ax.bar3d(xx_flat - bar_width * 0.5, yy_flat - bar_width * 0.5, 0, bar_width, bar_width, z_flat,
         color=color_flat,  # 颜色
         edgecolor="w",     # 白色描边
         shade=True,        # 加阴影
         alpha=0.95)

plt.xticks(x, lamda, rotation=-20)
plt.yticks(y, delta, rotation=20)
ax.tick_params(axis='z', which='major', labelsize=11, labelcolor='k')

ax.set_zticklabels([0.80, 0.82, 0.84, 0.86, 0.88, 0.90])

# 座标轴名
ax.set_xlabel("流形正则化系数", labelpad=5, fontsize=12)
ax.set_ylabel("源模型微调系数", labelpad=5, fontsize=12, loc='bottom')
ax.set_zlabel("平均F1-score", labelpad=5, fontsize=12)
plt.title("最优F1-score为{}".format(np.max(z1)), fontdict={'weight': 700, 'size': 15}, ha='center')

ax.set_zlim3d(0, 0.10)


# 座标轴范围
# ax.set_zlim(zmin=0, zmax=0.15)
cmap = plt.cm.get_cmap(name='rainbow')


# 保存
plt.tight_layout()
fig.savefig("bar3d_ssgan.png", bbox_inches='tight', pad_inches=0)
plt.close(fig)