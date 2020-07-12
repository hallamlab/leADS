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
plt.rc('font', **{'family': 'serif', 'serif': ['Times'], 'size': 30, 'weight': 'bold'})
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=30)
plt.rc('ytick', labelsize=30)
plt.rc('axes', labelsize=30)
plt.rc('legend', title_fontsize='large', fontsize=30, fancybox=True, framealpha=0.5, facecolor='None',
       edgecolor='black')
plt.rc('pdf', fonttype='TrueType')
plt.rc('grid', linestyle="dotted", color='grey')

## Data
labels_data = list()
labels = ['1', '2', '3', '5', '10', '15', '20', '50']
label_color = ['black', 'red', 'darkgreen', 'blue', 'maroon']

# entropy
labels_data.append([103.9265, 211.684, 404.8735, 756.6525, 
                    999.336, 1538.3495, 3080.2285, 5068.3185])
labels_data.append([0.226500000000001, 2.73400000000001, 81.9905,
                    32.0195, 41.416, 38.3415, 74.9325000000001,
                    88.9234999999999])

# mutual
labels_data.append([106.772, 209.903, 444.1385, 530.8155,
                    1013.2755, 1566.1975, 2103.2915,
                    4940.454])
labels_data.append([1.089, 1.111, 5.04049999999998,
                    4.72449999999998, 10.7465, 
                    24.5505000000001, 35.2755, 97.4090000000001])

# variation
labels_data.append([293.529, 643.573, 1241.658, 1990.0975,
                    4611.7555, 8725.5935, 8867.4905, 19114.6285])
labels_data.append([3.48000000000002, 6.11500000000001,
                    2.79399999999998, 17.9275, 226.6025, 314.9665,
                    134.6165, 583.2745])

# propensity
labels_data.append([115.7565, 344.382, 494.8475, 559.568, 
                    1124.264, 1956.7345, 2095.406, 6079.342])
labels_data.append([0.567500000000003, 3.38, 10.7185,
                    8.17500000000001, 19.9440000000001, 130.2035,
                    18.075, 361.217])

# random
labels_data.append([95.41, 97.5915, 99.546, 99.6365, 103.674,
                    106.4985, 93.103, 112.3575])
labels_data.append([0.795999999999999, 1.4885, 0.887, 1.3815,
                    1.692, 1.05950000000001, 1.771, 1.1415])

labels_data = np.array(labels_data) / 60

fig, ax = plt.subplots()
fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
plt.plot(labels, labels_data[0], '>--', color=label_color[0], markersize=15, lw=1, label='Entropy')
plt.fill_between(labels, labels_data[0] - labels_data[1], labels_data[0] + labels_data[1],
                 alpha=1, edgecolor='grey', facecolor='grey', linewidth=0)

plt.plot(labels, labels_data[2], '^--', color=label_color[0], markersize=15, lw=1, label='Mutual information')
plt.fill_between(labels, labels_data[2] - labels_data[3], labels_data[2] + labels_data[3],
                 alpha=1, edgecolor='grey', facecolor='grey', linewidth=0)

plt.plot(labels, labels_data[4], 's--', color=label_color[0], markersize=15, lw=1, label='Variation ratios')
plt.fill_between(labels, labels_data[4] - labels_data[5], labels_data[4] + labels_data[5],
                 alpha=1, edgecolor='grey', facecolor='grey', linewidth=0)

plt.plot(labels, labels_data[6], 'o--', color=label_color[0], markersize=15, lw=1, label='nPSP')
plt.fill_between(labels, labels_data[6] - labels_data[7], labels_data[6] + labels_data[7],
                 alpha=1, edgecolor='grey', facecolor='grey', linewidth=0)

plt.plot(labels, labels_data[7], 'X--', color=label_color[0], markersize=15, lw=1, label='Random')
plt.fill_between(labels, labels_data[7] - labels_data[8], labels_data[7] + labels_data[8],
                 alpha=1, edgecolor='grey', facecolor='grey', linewidth=0)

# plt.legend(loc="best")
plt.grid(True)

## Save as a .pdf file
ax.set_xlabel(r'$g$')
ax.set_ylabel('Time (min)')
#ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, fancybox=False, frameon=False, shadow=False)
fig.set_size_inches(8, 7)
fig.tight_layout()
fig.savefig(os.path.join(rspath, 'sizevstime.eps'), dpi=600)
