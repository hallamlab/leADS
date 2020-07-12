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
label_color = ['black', 'red', 'darkgreen', 'blue', 'maroon']
labels = [r'$30\%$', r'$50\%$', r'$70\%$']

## Data
labels_data.append([0.3380, 0.3413, 0.3632])
labels_data.append([0.3478, 0.3377, 0.3541])
labels_data.append([0.3411, 0.3382, 0.3423])
labels_data.append([0.3386, 0.3319, 0.3516])
labels_data.append([0.3069, 0.3312, 0.3352])
labels_data = np.array(labels_data) * 100

# plot
fig, ax = plt.subplots()
fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
plt.plot(labels, labels_data[0], '>--', color=label_color[0], markersize=15, lw=1, label='Entropy')
plt.plot(labels, labels_data[1], '^--', color=label_color[0], markersize=15, lw=1, label='Mutual information')
plt.plot(labels, labels_data[2], 's--', color=label_color[0], markersize=15, lw=1, label='Variation ratios')
plt.plot(labels, labels_data[3], 'o--', color=label_color[0], markersize=15, lw=1, label='nPSP')
plt.plot(labels, labels_data[4], 'X--', color=label_color[0], markersize=15, lw=1, label='Random')
plt.ylim(30, 38)
#plt.legend(loc="best")
plt.grid(True)

## Save as a .pdf file
ax.set_xlabel(r'per$\%$')
#ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, fancybox=False, frameon=False, shadow=False)
ax.set_ylabel(r'Average F1 Score ($\%$)')
fig.set_size_inches(8, 7)
fig.tight_layout()
fig.savefig(os.path.join(rspath, 'uncertainty.eps'), dpi=600)


def export_legend(ax, filename="legend.eps"):
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.axis('off')
    legend = ax2.legend(*ax.get_legend_handles_labels(), loc='upper center', 
                        ncol=3, fancybox=False, frameon=False, shadow=False)
    fig  = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.set_size_inches(14, 2)
    fig.savefig(os.path.join(rspath, filename), bbox_inches='tight')

export_legend(ax=ax)
