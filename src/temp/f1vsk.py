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
plt.rc('legend', title_fontsize='large', fontsize=25, fancybox=True, framealpha=0.5, facecolor='None',
       edgecolor='black')
plt.rc('pdf', fonttype='TrueType')
plt.rc('grid', linestyle="dotted", color='grey')

## Data
labels_data = list()
label_color = ['black', 'red', 'darkgreen', 'blue', 'maroon']
labels = ['5', '10', '15', '20', '30', '40', '50', '70', '90', '100']

## Data
labels_data.append([0.3541, 0.3468, 0.3611, 0.3585, 0.3502, 0.3322,
                    0.3411, 0.3517, 0.3483, 0.3439])  # Variation ratios
labels_data.append([0.3635, 0.3470, 0.3424, 0.3485, 0.3461, 0.3574, 
                    0.3386, 0.3422, 0.3354, 0.3427])  # nPSP
labels_data = np.array(labels_data) * 100

# plot
fig, ax = plt.subplots()
fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
plt.plot(labels, labels_data[0], 's--', color=label_color[0], markersize=15, lw=1, label='Variation ratios')
plt.plot(labels, labels_data[1], 'o--', color=label_color[0], markersize=15, lw=1, label='nPSP')
plt.ylim(28, 42)
#plt.legend(loc="best")
plt.grid(True)

## Save as a .pdf file
ax.set_xlabel(r'$k$')
#ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, fancybox=False, frameon=False, shadow=False)
ax.set_ylabel(r'Average F1 Score ($\%$)')
fig.set_size_inches(8, 7)
fig.tight_layout()
fig.savefig(os.path.join(rspath, 'f1vsk.eps'), dpi=600)
