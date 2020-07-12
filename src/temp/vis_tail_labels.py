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
rspath = os.path.join(REPO_PATH, 'result')
ospath = os.path.join(REPO_PATH, 'objectset')

# Data
file_name = 'biocyc21_tier23_9429'
samples_name = 'leADS_22_samples_final.pkl'
samples_name = 'leADS_07_samples_final.pkl'
y_name = file_name + '_y.pkl'
num_tails = 2
if num_tails <= 1:
    num_tails = 2

# load file
with open(os.path.join(dspath, y_name), mode='rb') as f_in:
    y = pkl.load(f_in)
with open(os.path.join(rspath, samples_name), mode='rb') as f_in:
    samples_name = pkl.load(f_in)


def tail_properties(y, f_name, num_tails):
    print('## Number of sample for {0}: {1}'.format(f_name, y.shape[0]))
    L_S = int(np.sum(y))
    LCard_S = L_S / y.shape[0]
    LDen_S = LCard_S / L_S
    DL_S = np.nonzero(np.sum(y, axis=0))[0].size
    PDL_S = DL_S / y.shape[0]
    print('\t1)- CLASS PROPERTIES...')
    print('\t\t>> Number of labels for {0}: {1}'.format(f_name, L_S))
    print('\t\t>> Label cardinality for {0}: {1:.4f}'.format(f_name, LCard_S))
    print('\t\t>> Label density for {0}: {1:.4f}'.format(f_name, LDen_S))
    print('\t\t>> Distinct label sets for {0}: {1}'.format(f_name, DL_S))
    print('\t\t>> Proportion of distinct label sets for {0}: {1:.4f}'.format(
        f_name, PDL_S))
    tail = np.sum(y, axis=0)
    tail = tail[np.nonzero(tail)[0]]
    tail[tail <= num_tails] = 1
    tail[tail > num_tails] = 0
    print('\t\t>> Number of tail labels of size {0}: {1}'.format(
        num_tails, int(tail.sum())))
    tail[tail == 0] = -1
    tail[tail == 1] = 0
    print('\t\t>> Number of dominant labels of size {0}: {1}'.format(
        num_tails + 1, int(np.count_nonzero(tail))))

    tail = np.sum(y, axis=0)
    tail = tail[np.nonzero(tail)[0]]
    ##### TODO: comment
    # with open(os.path.join(ospath, "biocyc.pkl"), mode='rb') as f_in:
    # data_object = pkl.load(f_in)
    # pathway_dict = data_object["pathway_id"]
    # pathway_common_names = [data_object['processed_kb']['metacyc'][5][pid][0][1]
    # for pid, pidx in pathway_dict.items()
    # if pid in data_object['processed_kb']['metacyc'][5]]
    # del data_object, pathway_dict
    # tmp = np.argsort(tail)
    # print(pathway_common_names[tmp[1]], pathway_common_names[tmp[-1]])
    # print(tail[tmp[1]], tail[tmp[-1]])
    #####

    tail = np.sort(tail)
    # print
    df_comp = pd.DataFrame(
        {"Label": np.arange(1, 1 + tail.shape[0]), "Sum": tail})

    # Prob bar
    alt.themes.enable('none')
    chart1 = alt.Chart(df_comp).properties(width=600, height=350).mark_bar(color="grey").encode(
        x=alt.X('Label:O', title="Pathway ID", sort='ascending'),
        y=alt.Y('Sum:Q', title="Number of Examples"))
    hrule = alt.Chart(df_comp).properties(width=600, height=350).mark_rule(
        color='red').encode(
        y=alt.Y('mean(Sum):Q', title=None, sort=None, axis=None))
    vrule = alt.Chart(df_comp).properties(width=600, height=350).mark_rule(
        color='red').encode(
        x=alt.X('mean(Label):Q', title=None, sort=None, axis=None))

    chart = (chart1).configure_header(
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
    chart.save(os.path.join(rspath, 'tails_bar.html'))


tail_properties(y=y.toarray(), f_name=file_name.lower(), num_tails=num_tails)
