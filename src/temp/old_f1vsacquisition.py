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
plt.rc('legend', title_fontsize='large', fontsize=24, fancybox=True, framealpha=0.5, facecolor='None',
       edgecolor='black')
plt.rc('pdf', fonttype='TrueType')
plt.rc('grid', linestyle="dotted", color='grey')
label_color = ['black', 'red', 'darkgreen', 'blue', 'maroon']

## Data
labels_data = list()
labels = ['H', 'M', 'V', 'nPSP']

## Data
labels_data = [51.60, 51.81, 52.84, 51.37]  # dependency
labels_data = np.array(labels_data)

# plot
fig, ax = plt.subplots()
fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
x_pos = np.arange(len(labels))
ax.bar(x_pos, labels_data, alpha=0.5, ecolor="black", capsize=20)
for idx, item in enumerate(x_pos):
    ax.text(item - 0.3, labels_data[idx] + 1, str(labels_data[idx]), fontsize=28)
plt.ylim(0, 60)
plt.grid(True)

## Save as a .pdf file
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_ylabel('Average F1 Score ($\%$)')
fig.set_size_inches(8, 6)
fig.tight_layout()
fig.savefig(os.path.join(rspath, 'f1vsacquisition1.eps'), dpi=600)

# data
labels_data = [39.16, 52.36, 52.99, 49.02]  # factorization
labels_data = np.array(labels_data)

# plot
fig, ax = plt.subplots()
fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
x_pos = np.arange(len(labels))
ax.bar(x_pos, labels_data, alpha=0.5, ecolor="black", capsize=20)
for idx, item in enumerate(x_pos):
    ax.text(item - 0.3, labels_data[idx] + 1, str(labels_data[idx]), fontsize=28)
plt.ylim(0, 60)
plt.grid(True)

## Save as a .pdf file
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_ylabel('Average F1 Score ($\%$)')
fig.set_size_inches(8, 6)
fig.tight_layout()
fig.savefig(os.path.join(rspath, 'f1vsacquisition2.eps'), dpi=600)
