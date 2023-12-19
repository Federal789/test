import matplotlib.pyplot as plt
import numpy as np

recall1 = np.array([0.81530266, 0.87784324, 0.90636634, 0.92325319, 0.93219245])
recall2 = np.array([0.75794627, 0.84893628, 0.88175555, 0.89894921, 0.91171503])

fig = plt.figure()

x = list(range(0, len(recall1)))
plt.plot(x, recall1, color='#800080', alpha=0.7, linestyle='--', linewidth=2, marker='X',
         markeredgecolor='#dd85d7', markersize='8', markeredgewidth=3, label='WA-SSGAN')
plt.plot(x, recall2, color='#3366FF', alpha=0.9, linestyle='--', linewidth=2, marker='o',
         markeredgecolor='#00FF00', markersize='9', markeredgewidth=3, label='CNN')
plt.legend(loc='lower right', borderpad=0.8, fancybox=True)

ymin, ymax = 0.725, 0.95

def aax(x):
    x0 = (x + 0.2) / (3.5 + 0.2)
    return x0

def aay(y):
    y0 = (y-ymin) / (ymax - ymin)
    return y0

plt.grid(alpha=0.5, linestyle='-.', linewidth=1.5, axis='y')
ax2 = plt.subplot(111)
ax2.set_ylim([ymin, ymax])
ax2.set_xlim([-0.2, len(x)-0.5])
xlabel = [1, 3, 5, 7, 9]
plt.xticks(range(len(recall1)), labels=xlabel, fontsize=10, fontweight=500)
plt.yticks(fontsize=10, fontweight=500)

# plt.axhline(y=recall1[0], xmin=aax(0), xmax=aax(1), ls='--', lw=2, c='k')
# plt.axhline(y=recall1[1], xmin=aax(1), xmax=aax(2), ls='--', lw=2, c='k')
# plt.axhline(y=recall1[2], xmin=aax(2), xmax=aax(3), ls='--', lw=2, c='k')
plt.axvline(x=0, ymin=aay(recall2[0]), ymax=aay(recall1[0]), ls='--', lw=1, c='#FF4500')
plt.axvline(x=1, ymin=aay(recall2[1]), ymax=aay(recall1[1]), ls='--', lw=1, c='#FF4500')
plt.axvline(x=2, ymin=aay(recall2[2]), ymax=aay(recall1[2]), ls='--', lw=1, c='#FF4500')
plt.axvline(x=3, ymin=aay(recall2[3]), ymax=aay(recall1[3]), ls='--', lw=1, c='#FF4500')
plt.axvline(x=4, ymin=aay(recall2[4]), ymax=aay(recall1[4]), ls='--', lw=1, c='#FF4500')

for a, b in zip(x, recall1):
    plt.text(a+0.05, b+0.012, '%.2f' % b, ha='center', va='top', fontdict={'size': 10.5, 'weight': 600, 'family':'Arial'})
for a, b in zip(x, recall2):
    plt.text(a+0.05, b-0.008, '%.2f' % b, ha='center', va='top', fontdict={'size': 10.5, 'weight': 600, 'family':'Arial'})
plt.xlabel('Label ratio(%)', fontdict={'weight': 550, 'size': 11, 'family': 'Arial'})
plt.ylabel('Average F1-score', fontdict={'weight': 550, 'size': 11, 'family': 'Arial'})
#plt.title('The average F1-score of WA-SSGAN and discriminator', fontdict={'weight': 700, 'size': 13, 'family':'Arial'})

plt.tight_layout()
fig.savefig("CNN-SSGAN.png", bbox_inches='tight', pad_inches=0)
plt.close(fig)