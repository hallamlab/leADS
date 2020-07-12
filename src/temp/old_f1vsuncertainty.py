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
plt.rc('legend', title_fontsize='large', fontsize=18, fancybox=True, framealpha=0.5, facecolor='None',
       edgecolor='black')
plt.rc('pdf', fonttype='TrueType')
plt.rc('grid', linestyle="dotted", color='grey')

## Data
labels_data = list()
label_color = ['black', 'red', 'darkgreen', 'blue', 'maroon']
labels = [r'$30\%$', r'$50\%$', r'$70\%$']

## Data
labels_data.append([52.84, 54.67, 56.32])  # dep
labels_data.append([52.12, 55.70, 56.21])  # fact
labels_data.append([55.91, 58.41, 57.80])  # random
labels_data = np.array(labels_data)

# plot
fig, ax = plt.subplots()
fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
plt.plot(labels, labels_data[0], 's--', color=label_color[0], markersize=15, lw=1, label='Dependency')
plt.plot(labels, labels_data[1], 'o--', color=label_color[0], markersize=15, lw=1, label='Factorization')
plt.plot(labels, labels_data[2], 'X--', color=label_color[0], markersize=15, lw=1, label='Random')
plt.ylim(51, 60)
# plt.legend(loc="best")
plt.grid(True)

## Save as a .pdf file
# ax.set_xlabel(r'per$\%$')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3, fancybox=False, frameon=False, shadow=False)
ax.set_ylabel(r'Average F1 Score ($\%$)')
fig.set_size_inches(8, 7)
fig.tight_layout()
fig.savefig(os.path.join(rspath, 'uncertainty.eps'), dpi=600)
