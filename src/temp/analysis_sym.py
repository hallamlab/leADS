import altair as alt
import numpy as np
import os
import pandas as pd
import pickle as pkl

os.system('clear')

selected_pathways = ['PWY-6163', 'ARO-PWY', 'PWY-6165', 'PHESYN', 'PWY-3462', 'PWY-7432',
                     'TRPSYN-PWY',
                     'PWY-5686', 'PWY-7790', 'PWY-7791',
                     'VALSYN-PWY',
                     'LEUSYN-PWY',
                     'DAPLYSINESYN-PWY', 'PWY-2941', 'PWY-2942', 'LYSINE-AMINOAD-PWY', 'PWY-3081', 'PWY-5097',
                     'HOMOSER-THRESYN-PWY',
                     'ILEUSYN-PWY', 'PWY-5101', 'PWY-5103', 'PWY-5104', 'PWY-5108',
                     'HISTSYN-PWY', 'PWY-5029',
                     'HOMOSER-METSYN-PWY', 'PWY-702', 'HSERMETANA-PWY', 'PWY-7977']
selected_topologies = ['Phenylalanine', 'Phenylalanine', 'Phenylalanine', 'Phenylalanine', 'Phenylalanine',
                       'Phenylalanine',
                       'Tryptophan',
                       'Arginine', 'Arginine', 'Arginine',
                       'Valine',
                       'Leucine',
                       'Threonine', 'Threonine', 'Threonine', 'Threonine', 'Threonine', 'Threonine', 'Threonine',
                       'Isoleucine', 'Isoleucine', 'Isoleucine', 'Isoleucine', 'Isoleucine',
                       'Histidine', 'Histidine',
                       'Methionine', 'Methionine', 'Methionine', 'Methionine']
selected_pathways = ['PHESYN', 'TRPSYN-PWY', 'ARGSYNBSUB-PWY', 'VALSYN-PWY',
                     'LEUSYN-PWY', 'DAPLYSINESYN-PWY', 'HOMOSER-THRESYN-PWY',
                     'ILEUSYN-PWY', 'HISTSYN-PWY', 'HOMOSER-METSYN-PWY']
selected_topologies = ['Phenylalanine', 'Tryptophan', 'Arginine', 'Valine', 'Leucine',
                       'Threonine', 'Threonine', 'Isoleucine', 'Histidine', 'Methionine']

## Paths for the required files
DIRECTORY_PATH = os.getcwd()
REPO_PATH = DIRECTORY_PATH.split('/')
REPO_PATH = '/'.join(REPO_PATH[:-3])
rspath = os.path.join(REPO_PATH, 'result')
dspath = os.path.join(REPO_PATH, 'dataset')

## file name
folder = "leADS_4_symbionts"
save_name = 'leADS_4_voting'

comp_coverage = True
epsilon = 0.05
if comp_coverage:
    epsilon = 0
y = "symbionts_y.pkl"
y_biocyc = "biocyc_y.pkl"
vocab = 'vocab_biocyc.pkl'
tmp = os.path.join('symbionts/leads/', folder)
comp_file_name = os.path.join(tmp, 'collected_pathway_report.csv')
abd_file_name = os.path.join(tmp, 'collected_pathway_abd.csv')
cov_file_name = os.path.join(tmp, 'collected_pathway_cov.csv')

header = "2"
ds_columns = {"0": "Moranella", "1": "Tremblaya", "2": "Combined"}
abd_columns = {"0": "Abundance Moranella", "1": "Abundance Tremblaya", "2": "Abundance Combined"}
cov_columns = {"0": "Coverage Moranella", "1": "Coverage Tremblaya", "2": "Coverage Combined"}

## load data
df = pd.read_csv(os.path.join(dspath, comp_file_name), sep='\t')
df = df[['Sample_ID', 'PathwayFrameID', 'Status']]
df.index.names = ['Sample_ID']
df_abd = pd.read_csv(os.path.join(dspath, abd_file_name), sep='\t')
df_cov = pd.read_csv(os.path.join(dspath, cov_file_name), sep='\t')
with open(os.path.join(dspath, vocab), mode='rb') as f_in:
    vocab = pkl.load(f_in)
with open(os.path.join(dspath, y), mode='rb') as f_in:
    y = pkl.load(f_in)
with open(os.path.join(dspath, y_biocyc), mode='rb') as f_in:
    y_biocyc = pkl.load(f_in)

## constraint based on these pathways
y_biocyc = y_biocyc.sum(0)
y_biocyc = np.nonzero(y_biocyc)[1]
y_biocyc = [vocab[int(ptwy)] for ptwy in y_biocyc]
tmp = [selected_pathways.index(ptwy) for ptwy in selected_pathways if ptwy in y_biocyc]
selected_pathways = list(np.array(selected_pathways)[tmp])
selected_topologies = list(np.array(selected_topologies)[tmp])

## data processing
common_name = list(df_abd['PathwaysCommonName'])
df_comp = pd.DataFrame(data=np.zeros((len(vocab.keys()), 3)), index=list(vocab.values()),
                       columns=tuple(np.arange(3)))
df_comp.index.names = ['PathwayFrameID']
for idx in np.arange(y.shape[0]):
    model_pathways = list(zip(df.loc[df['Sample_ID'] == idx, 'PathwayFrameID'].tolist(),
                              df.loc[df['Sample_ID'] == idx, 'Status'].tolist()))
    model_pathways = [ptwy for ptwy, status in model_pathways if int(status) != 0]
    pathways = np.nonzero(y[idx])[1]
    pathways = [vocab[int(ptwy)] for ptwy in pathways]
    for path in pathways:
        df_comp[idx].loc[path] += 1
    for path in model_pathways:
        df_comp[idx].loc[path] += 2
df_comp.insert(loc=0, column='PathwaysCommonName', value=common_name)
df_comp = df_comp.reset_index(drop=False)
tmp = [selected_pathways.index(item) for idx, item in enumerate(df_comp.iloc[:, 0]) if item in selected_pathways]
selected_pathways = np.array(selected_pathways)[tmp]
selected_topologies = np.array(selected_topologies)[tmp]
list_pathways_idx = [df_comp[df_comp['PathwayFrameID'] == item].index.values.astype(int)[0] for item in
                     selected_pathways]
df_comp = df_comp.iloc[list_pathways_idx, :]
df_comp = df_comp.reset_index(drop=True)
list_pathways_idx = [df_abd[df_abd['PathwayFrameID'] == item].index.values.astype(int)[0] for item in selected_pathways]
df_abd = df_abd.iloc[list_pathways_idx, :]
df_abd = df_abd.reset_index(drop=True)
list_pathways_idx = [df_cov[df_cov['PathwayFrameID'] == item].index.values.astype(int)[0] for item in selected_pathways]
df_cov = df_cov.iloc[list_pathways_idx, :]
df_cov = df_cov.reset_index(drop=True)

## add a column "Amino Acid"
new_series = pd.Series([])
for i in range(len(df_comp)):
    new_series[i] = selected_topologies[list(selected_pathways).index(df_comp.iloc[i, 0])]
df_abd.insert(5, "Amino Acid", new_series)
df_cov.insert(5, "Amino Acid", new_series)
df_comp.insert(5, "Amino Acid", new_series)

# drop "Pathways_ID"
df_abd.drop("PathwayFrameID", axis=1, inplace=True)
df_cov.drop("PathwayFrameID", axis=1, inplace=True)
df_comp.drop("PathwayFrameID", axis=1, inplace=True)
for i in range(len(df_comp)):
    for k in abd_columns.keys():
        if k == header:
            continue
        df_abd[k][i] = df_abd[k][i] * df_cov[k][i] + epsilon
        df_abd[header][i] = df_abd[header][i] + df_abd[k][i]

# remove non-coverage pathways
if comp_coverage:
    remove_idx = [i for i in range(len(df_cov)) if np.sum(df_cov.loc[i][1:4]) == 0]
    df_abd.drop(remove_idx, inplace=True)
    df_cov.drop(remove_idx, inplace=True)
    df_comp.drop(remove_idx, inplace=True)

# rename columns
df_abd.rename(columns=dict((int(k), v) for k, v in abd_columns.items()), inplace=True)
df_cov.rename(columns=dict((int(k), v) for k, v in cov_columns.items()), inplace=True)
df_comp.rename(columns=dict((int(k), v) for k, v in ds_columns.items()), inplace=True)

# melt and merge
df_abd = df_abd.melt(['PathwaysCommonName', "Amino Acid"], var_name='AbundanceDataSet', value_name='Abundance')
df_comp = df_comp.melt(['PathwaysCommonName', "Amino Acid"], var_name='DataSet', value_name='Status')
df_comp = pd.concat([df_comp, df_abd.iloc[:, 3:]], axis=1, sort=False)

# add a column "Predicted"
new_series = pd.Series([])
no_one = 0
patho = 0
ml = 0
both = 0
for i in range(len(df_comp)):
    if int(df_comp["Status"][i]) == 0:
        new_series[i] = "None"
        no_one += 1
    elif int(df_comp["Status"][i]) == 1:
        new_series[i] = "PathoLogic"
        patho += 1
    elif int(df_comp["Status"][i]) == 2:
        new_series[i] = "leADS"
        ml += 1
    else:
        new_series[i] = "Both"
        both += 1
print("None: {0}, PathoLogic: {1}, leADS: {2}, Both: {3}".format(no_one, patho, ml, both))
df_comp.insert(4, "Predicted", new_series)

## plot
chart = alt.Chart(df_comp).properties(width=80, height=450).mark_circle(opacity=1).encode(
    x=alt.X('DataSet:O', title=None, sort=None),
    y=alt.Y('PathwaysCommonName:O', title='MetaCyc Pathways', sort=None),
    color='Predicted:N',
    size=alt.Size('Abundance', scale=alt.Scale(domain=[0, 1]))).facet(
    column=alt.Column('Amino Acid:N')).configure_header(
    # titleColor='green',
    titleFontSize=20,
    labelColor='red',
    labelFontSize=15
).configure_axis(
    titleFontSize=20,
    labelLimit=500,
    labelFontSize=20,
    labelPadding=5,
    grid=True).configure_legend(
    strokeColor='gray',
    fillColor='#EEEEEE',
    padding=10,
    cornerRadius=10).configure_axisY(
    titleAngle=0,
    titleY=-10,
    titleX=-120).resolve_scale(
    x='independent')

## save
chart.save(os.path.join(rspath, save_name + '.html'))

df_comp["Amino Acid"]
for i in range(len(df_comp)):
    df_comp["PathwaysCommonName"][i] = df_comp["PathwaysCommonName"][i] + " [" + df_comp["Amino Acid"][i] + "]"
df_comp.sort_values(by=['Amino Acid'], inplace=True)

chart = alt.Chart(df_comp).properties(width=120, height=280).mark_circle(opacity=1).encode(
    x=alt.X('DataSet:O', title=None, sort="descending"),
    y=alt.Y('PathwaysCommonName:O', title='MetaCyc Pathways', sort=None),
    color='Predicted:N',
    size=alt.Size('Abundance', scale=alt.Scale(domain=[0, 1]))).configure_header(
    titleFontSize=20,
    labelColor='red',
    labelFontSize=15
).configure_axis(
    titleFontSize=20,
    labelLimit=500,
    labelFontSize=20,
    labelPadding=5,
    grid=True).configure_legend(
    strokeColor='gray',
    fillColor='white',
    padding=10,
    cornerRadius=10).configure_axisY(
    titleAngle=0,
    titleY=-10,
    titleX=-120).resolve_scale(
    x='independent')

## save
chart.save(os.path.join(rspath, save_name + '_compact.html'))
