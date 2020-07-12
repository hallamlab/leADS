import altair as alt
import numpy as np
import os
import pandas as pd
import pickle as pkl

os.system('clear')

# Paths for the required files
DIRECTORY_PATH = os.getcwd()
REPO_PATH = DIRECTORY_PATH.split('/')
REPO_PATH = '/'.join(REPO_PATH[:-3])
dspath = os.path.join(REPO_PATH, 'dataset/biocyc/biocyc21')
rspath = os.path.join(REPO_PATH, 'result/leads')

# Data
file_name = 'leADS_pp'  # 'leADS_ey' (4752 samples and 1380 distinct pathways), 'leADS_ml' (4762 samples and 1378 distinct pathways), 
                        # 'leADS_vr' (5506 samples and 1431 distinct pathways), 'leADS_pp' (4752 samples and 1404 distinct pathways)
y_name = 'biocyc21_tier23_9429_y.pkl'
num_tails = 2
if num_tails <= 1:
    num_tails = 2

# load file
with open(os.path.join(dspath, y_name), mode='rb') as f_in:
    y = pkl.load(f_in)
with open(os.path.join(rspath, file_name + '_samples.pkl'), mode='rb') as f_in:
    samples_name = pkl.load(f_in)


def tail_properties(y, f_name, num_tails):
    L_S = int(np.sum(y))
    LCard_S = L_S / y.shape[0]
    LDen_S = LCard_S / L_S
    DL_S = np.nonzero(np.sum(y, axis=0))[0].size
    PDL_S = DL_S / y.shape[0]
    print('1)- DATA PROPERTIES for {0}...'.format("biocyc"))
    print('\t>> Number of samples: {0}'.format(y.shape[0]))
    print('\t>> Number of labels: {0}'.format(L_S))
    print('\t>> Label cardinality: {0:.4f}'.format(LCard_S))
    print('\t>> Label density: {0:.4f}'.format(LDen_S))
    print('\t>> Distinct label sets: {0}'.format(DL_S))
    print('\t>> Proportion of distinct label sets: {0:.4f}'.format(PDL_S))
    tail = np.sum(y, axis=0)
    tail = tail[np.nonzero(tail)[0]]
    tail[tail <= num_tails] = 1
    tail[tail > num_tails] = 0
    print('\t>> Number of tail labels of size {0}: {1}'.format(
        num_tails, int(tail.sum())))
    tail[tail == 0] = -1
    tail[tail == 1] = 0
    print('\t>> Number of dominant labels of size {0}: {1}'.format(
        num_tails + 1, int(np.count_nonzero(tail))))

    tail = np.sum(y, axis=0)
    ntail_idx = np.nonzero(tail)[0]
    tail = tail[ntail_idx]
    tail_idx = np.argsort(tail)
    tail = tail[tail_idx]

    y = y[samples_name]
    tail_leads = np.sum(y, axis=0)
    tail_leads = tail_leads[ntail_idx]
    tail_leads = tail_leads[tail_idx]

    L_S = int(np.sum(y))
    LCard_S = L_S / y.shape[0]
    LDen_S = LCard_S / L_S
    DL_S = np.nonzero(np.sum(y, axis=0))[0].size
    PDL_S = DL_S / y.shape[0]
    print('2)- DATA PROPERTIES for {0}...'.format(f_name))
    print('\t>> Number of samples: {0}'.format(y.shape[0]))
    print('\t>> Number of labels: {0}'.format(L_S))
    print('\t>> Label cardinality: {0:.4f}'.format(LCard_S))
    print('\t>> Label density: {0:.4f}'.format(LDen_S))
    print('\t>> Distinct label sets: {0}'.format(DL_S))
    print('\t>> Proportion of distinct label sets: {0:.4f}'.format(PDL_S))

    # print
    df_comp = pd.DataFrame(
        {"Label": np.arange(1, 1 + tail.shape[0]), "All": tail, "leADS": tail_leads})
    df_comp = df_comp.melt(['Label'], var_name='Subsampling', value_name='Sum')

    # Prob bar
    alt.themes.enable('none')
    chart = alt.Chart(df_comp).properties(width=600, height=350).mark_area(color="grey").encode(
        x=alt.X('Label:O', title="Pathway ID", sort='ascending'),
        y=alt.Y('Sum:Q', title="Number of Examples"),
        color=alt.Color('Subsampling:N', scale=alt.Scale(
            range=['grey', 'black'])),
    ).configure_header(
        titleFontSize=20,
        labelFontSize=15
    ).configure_axis(
        labelLimit=500,
        titleFontSize=20,
        labelFontSize=12,
        labelPadding=5,
    ).configure_axisY(
        grid=False
    ).configure_legend(
        strokeColor='gray',
        fillColor='white',
        padding=10,
        cornerRadius=10).resolve_scale(x='independent')
    # save
    chart.save(os.path.join(rspath, f_name.lower() + '_tails' + '.html'))


tail_properties(y=y.toarray(), f_name=file_name, num_tails=num_tails)
