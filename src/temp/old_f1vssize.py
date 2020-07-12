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
labels_data = list()
labels = ['1', '3', '5', '10', '15', '20']
label_color = ['black', 'red', 'darkgreen', 'blue', 'maroon']
labels_data.append([0.3304, 0.4136, 0.4426, 0.4752, 0.4778, 0.4767])
labels_data.append([0.5824, 0.5403, 0.5318, 0.4728, 0.4870, 0.4681])
labels_data.append([0.4972, 0.5315, 0.5233, 0.4798, 0.4907, 0.4635])
labels_data = np.array(labels_data) * 100

fig, ax = plt.subplots()
fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
plt.plot(labels, labels_data[0], 's--', color=label_color[0], markersize=15, lw=1, label='pref-vrank (fact)')
plt.plot(labels, labels_data[1], 'o--', color=label_color[0], markersize=15, lw=1, label='pref-voting (fact)')
plt.plot(labels, labels_data[2], 'X--', color=label_color[0], markersize=15, lw=1, label='pref-voting (dep)')
plt.ylim(30, 65)
# plt.legend(loc="best")
plt.grid(True)

## Save as a .pdf file
ax.set_xlabel(r'$g$')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3, fancybox=False, frameon=False, shadow=False)
ax.set_ylabel(r'Average F1 Score ($\%$)')
fig.set_size_inches(8, 7)
fig.tight_layout()
fig.savefig(os.path.join(rspath, 'sizevsf1.eps'), dpi=600)
