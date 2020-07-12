import matplotlib.pyplot as plt
import numpy as np
import os

os.system('clear')

## Paths for the required files
DIRECTORY_PATH = os.getcwd()
REPO_PATH = DIRECTORY_PATH.split('/')
REPO_PATH = '/'.join(REPO_PATH[:-3])
rspath = os.path.join(REPO_PATH, 'result')

## Common Information for plots
plt.rc('font', **{'family': 'serif', 'serif': ['Times'], 'size': 20, 'weight': 'bold'})
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=30)
plt.rc('ytick', labelsize=30)
plt.rc('axes', labelsize=30)
plt.rc('legend', title_fontsize='large', fontsize=16, fancybox=True, framealpha=0.5, facecolor='None',
       edgecolor='black')
plt.rc('pdf', fonttype='TrueType')
plt.rc('grid', linestyle="dotted", color='grey')

## Data
labels = ['1', '3', '5', '10', '15', '20']
label_color = ['black', 'red', 'darkgreen', 'blue', 'maroon']
labels_data = list()
# dependency
# 22	leADS_28_1_cost.txt	    1587.99	9.076999999999998	1.6431713825146084	0.14475341793561958
# 24	leADS_29_1_cost.txt	    4737.649	11.125	1.6703498031678858	0.04852239865280805
# 27	leADS_30_1_cost.txt	    7871.4455	28.842499999999745	1.5527444409289726	0.057788035229320855
# 29	leADS_31_1_cost.txt	    16088.3425	520.4145000000008	1.5629289579024235	0.03236408984910921
# 31	leADS_32_1_cost.txt	    25074.764499999997	704.3894999999993	1.5401368810798974	0.010372209203539318
# 33	leADS_33_1_cost.txt	    31640.2605	216.29449999999997	1.5982932717498257	0.025862242165847427
labels_data.append([1587.99, 4737.649, 7871.4455, 16088.3425, 25074.764499999997, 31640.2605])
labels_data.append(
    [9.076999999999998, 11.125, 28.842499999999745, 520.4145000000008, 704.3894999999993, 216.29449999999997])

# factorization
# 23	leADS_28_cost.txt	476.5615	1.433500000000009	1.647403200331822	0.1575560962365089
# 25	leADS_29_cost.txt	1383.6725000000001	19.934499999999957	1.5454595942636888	0.0021681049567288113
# 28	leADS_30_cost.txt	2221.242	10.837999999999965	1.597445227692731	0.005357109394504511
# 30	leADS_31_cost.txt	4274.3724999999995	35.36349999999993	1.6005140778245532	0.027716013877872703
# 32	leADS_32_cost.txt	6389.2085	99.27750000000015	1.6365079162190561	0.01150523602695075
# 34	leADS_33_cost.txt	8421.978	88.89499999999953	1.6338416078535916	0.006632529389335451
labels_data.append([476.5615, 1383.6725000000001, 2221.242, 4274.3724999999995, 6389.2085, 8421.978])
labels_data.append([1.433500000000009, 19.934499999999957, 10.837999999999965, 35.36349999999993, 99.27750000000015,
                    88.89499999999953])
labels_data = np.array(labels_data) / 60

fig, ax = plt.subplots()
fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
plt.plot(labels, labels_data[0], 'X--', color=label_color[0], markersize=15, lw=1, label='Dependency')
plt.fill_between(labels, labels_data[0] - labels_data[1], labels_data[0] + labels_data[1],
                 alpha=1, edgecolor='grey', facecolor='grey', linewidth=0)

plt.plot(labels, labels_data[2], 'o--', color=label_color[0], markersize=15, lw=1, label='Factorization')
plt.fill_between(labels, labels_data[2] - labels_data[3], labels_data[2] + labels_data[3],
                 alpha=1, edgecolor='grey', facecolor='grey', linewidth=0)
plt.ylim(-20, 600)
# plt.legend(loc="best")
plt.grid(True)

## Save as a .pdf file
ax.set_xlabel(r'$g$')
ax.set_ylabel('Time (min)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, fancybox=False, frameon=False, shadow=False)
fig.set_size_inches(8, 7)
fig.tight_layout()
fig.savefig(os.path.join(rspath, 'sizevstime.eps'), dpi=600)
