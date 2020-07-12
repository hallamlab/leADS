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

labels_data.append([0.2630, 0.3170, 0.3380, 0.3797, 0.4877, 0.5354, 0.5589, 0.5876])
labels_data.append([0.2467, 0.3163, 0.3478, 0.3775, 0.4786, 0.5434, 0.5744, 0.5918])
labels_data.append([0.2467, 0.3218, 0.3411, 0.3912, 0.4689, 0.5504, 0.5731, 0.5880])
labels_data.append([0.2493, 0.3035, 0.3386, 0.3864, 0.4877, 0.5187, 0.5616, 0.5865])
labels_data.append([0.2614, 0.3000, 0.2839, 0.3351, 0.3095, 0.4156, 0.4533, 0.4587])

labels_data = np.array(labels_data) * 100

fig, ax = plt.subplots()
fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
plt.plot(labels, labels_data[0], '>--', color=label_color[0], markersize=15, lw=1, label='Entropy')
plt.plot(labels, labels_data[1], '^--', color=label_color[0], markersize=15, lw=1, label='Mutual information')
plt.plot(labels, labels_data[2], 's--', color=label_color[0], markersize=15, lw=1, label='Variation ratios')
plt.plot(labels, labels_data[3], 'o--', color=label_color[0], markersize=15, lw=1, label='nPSP')
plt.plot(labels, labels_data[4], 'X--', color=label_color[0], markersize=15, lw=1, label='Random')
plt.ylim(22, 65)
# plt.legend(loc="best")
plt.grid(True)

## Save as a .pdf file
ax.set_xlabel(r'$g$')
#ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3, fancybox=False, frameon=False, shadow=False)
ax.set_ylabel(r'Average F1 Score ($\%$)')
fig.set_size_inches(8, 7)
fig.tight_layout()
fig.savefig(os.path.join(rspath, 'sizevsf1.eps'), dpi=600)
