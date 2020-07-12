import os
import pickle as pkl

import altair as alt
import numpy as np
import pandas as pd

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

# Paths for the required files
DIRECTORY_PATH = os.getcwd()
REPO_PATH = DIRECTORY_PATH.split('/')
REPO_PATH = '/'.join(REPO_PATH[:-3])
rspath = os.path.join(REPO_PATH, 'result')
dspath = os.path.join(REPO_PATH, 'dataset')
mllr_dspath = os.path.join(dspath, 'symbionts/mllr')
ospath = os.path.join(REPO_PATH, 'objectset')

# File name
sample_idx = 1
save_name = 'leads'
folders_name = "symbionts"
abd_file_name = 'mg_symbionts_abd.csv'
folders = ['pathologic', 'mllr', 'triumpf']
ds_names = ['PathoLogic', 'mlLGPR', 'triUMPF']
files = ["symbionts_pathologic_y.pkl", "symbionts_mllr_y.pkl", "triUMPF_e_symbionts_y.pkl"]
leads_names = ['Random', 'Full', 'Entropy', 'Mutual information', 'Variation ratios', 'nPSP']
leads_files = ["leADS_ra_symbionts_y.pkl", "leADS_fl_symbionts_y.pkl", 
               "leADS_ey_symbionts_y.pkl", "leADS_ml_symbionts_y.pkl", 
              "leADS_vr_symbionts_y.pkl", "leADS_pp_symbionts_y.pkl"]
ds_columns = {"nc_015735cyc": "Moranella",
              "nc_015736cyc": "Tremblaya", "symcombinedcyc": "Combined"}
tmp = list(ds_columns.keys())[sample_idx]
ds_columns = {tmp: ds_columns[tmp]}
save_name = save_name + "_" + list(ds_columns.values())[0]

# Load data
with open(os.path.join(dspath, 'vocab_biocyc.pkl'), mode='rb') as f_in:
    vocab = pkl.load(f_in)
with open(os.path.join(ospath, "biocyc.pkl"), mode='rb') as f_in:
    data_object = pkl.load(f_in)
pathway_dict = data_object["pathway_id"]
pathway_common_names = dict((pidx, data_object['processed_kb']['metacyc'][5][pid][0][1])
                            for pid, pidx in pathway_dict.items()
                            if pid in data_object['processed_kb']['metacyc'][5])
pathway_dict = [k for k, v in sorted(
    pathway_dict.items(), key=lambda item: item[1])]
pathway_common_names = [v for k, v in sorted(
    pathway_common_names.items(), key=lambda item: item[0])]
pathway_idx = np.array([idx for idx, k in enumerate(
    pathway_dict) if k in selected_pathways])
pathway_dict = np.array(pathway_dict)[pathway_idx]
pathway_common_names = np.array(pathway_common_names)[pathway_idx]
selected_topologies = np.array([selected_topologies[selected_pathways.index(
    ptwy)] for ptwy in pathway_dict if ptwy in selected_pathways])
del data_object
abd = pd.read_csv(os.path.join(mllr_dspath, abd_file_name), sep='\t')
abd = abd.iloc[:, [0, 1, 2 + sample_idx]]

# Data processing
tmp = [abd[abd['Pathways_ID'] == item].index.values.astype(
    int)[0] for item in pathway_dict]
abd = abd.iloc[tmp, :]
abd = np.array(abd.iloc[:, 2])

# all models
df_comp = pd.DataFrame()
predicted = list()
for idx, f in enumerate(files):
    f = os.path.join(dspath, folders_name, folders[idx], f)
    with open(f, mode='rb') as f_in:
        tmp = pkl.load(f_in)
        tmp = tmp[sample_idx]
        tmp = tmp[:, pathway_idx]
        predicted.append(
            ["Yes" if item == 1 else "No" for item in list(tmp.toarray()[0])])
        df_comp.insert(loc=idx, column=ds_names[idx], value=abd)
predicted = np.concatenate(predicted)
df_comp.insert(loc=0, column="PathwaysCommonName", value=pathway_common_names)
df_comp.insert(loc=1, column="Amino Acid", value=selected_topologies)
df_comp = df_comp.reset_index(drop=True)

for i in range(len(df_comp)):
    df_comp["PathwaysCommonName"][i] = df_comp["PathwaysCommonName"][i] + \
        " [" + df_comp["Amino Acid"][i] + "]"

df_comp = df_comp.melt(['PathwaysCommonName', 'Amino Acid'],
                       var_name='DataSet', value_name='Abundance')
df_comp.insert(df_comp.shape[1], "Predicted", predicted)
df_comp.sort_values(by=['Amino Acid'], inplace=True)
df_comp.drop("Amino Acid", axis=1, inplace=True)
remove_idx = [i for i in range(len(df_comp)) if np.sum(df_comp.loc[i][2]) == 0]
df_comp.drop(remove_idx, inplace=True)

# mlts
df_leads = pd.DataFrame()
predicted = list()
for idx, f in enumerate(leads_files):
    f = os.path.join(dspath, folders_name, 'leads', f)
    with open(f, mode='rb') as f_in:
        tmp = pkl.load(f_in)
        tmp = tmp[sample_idx]
        tmp = tmp[:, pathway_idx]
        predicted.append(
            ["Yes" if item == 1 else "No" for item in list(tmp.toarray()[0])])
        df_leads.insert(loc=idx, column=leads_names[idx], value=abd)
predicted = np.concatenate(predicted)
df_leads.insert(loc=0, column="PathwaysCommonName", value=pathway_common_names)
df_leads.insert(loc=1, column="Amino Acid", value=selected_topologies)
df_leads = df_leads.reset_index(drop=True)

for i in range(len(df_leads)):
    df_leads["PathwaysCommonName"][i] = df_leads["PathwaysCommonName"][i] + \
        " [" + df_leads["Amino Acid"][i] + "]"

df_leads = df_leads.melt(['PathwaysCommonName', 'Amino Acid'],
                       var_name='DataSet', value_name='Abundance')
df_leads.insert(df_leads.shape[1], "Predicted", predicted)
df_leads.sort_values(by=['Amino Acid'], inplace=True)
df_leads.drop("Amino Acid", axis=1, inplace=True)
remove_idx = [i for i in range(len(df_leads)) if np.sum(df_leads.loc[i][2]) == 0]
df_leads.drop(remove_idx, inplace=True)

# Plot
chart1 = alt.Chart(df_comp).properties(width=100, height=325).mark_circle(opacity=1).encode(
    x=alt.X('DataSet:O', title=None, sort=ds_names),
    y=alt.Y('PathwaysCommonName:O', title='MetaCyc Pathways', sort=None),
    color=alt.Color('Predicted:N', scale=alt.Scale(range=['grey', 'black'])),
    size=alt.Size('Abundance', scale=alt.Scale(domain=[0, 1])))

chart2 = alt.Chart(df_leads).properties(width=200, height=325).mark_circle(opacity=1).encode(
    x=alt.X('DataSet:O', title=None, sort=leads_names),
    y=alt.Y('PathwaysCommonName:O', title=None, sort=None, axis=None),
    color=alt.Color('Predicted:N', scale=alt.Scale(range=['grey', 'black'])),
    size=alt.Size('Abundance', scale=alt.Scale(domain=[0, 1])))

chart = (chart1 | chart2).configure_header(
    titleFontSize=20,
    labelColor='red',
    labelFontSize=15
).configure_axis(
    titleFontSize=20,
    labelLimit=500,
    labelFontSize=20,
    labelPadding=5,
    grid=False).configure_legend(
    strokeColor='gray',
    fillColor='white',
    padding=10,
    cornerRadius=10
).configure_axisY(
    labelFontSize=15,
    titleAngle=0,
    titleY=-10,
    titleX=-120
).resolve_scale(x='independent')

# save
chart.save(os.path.join(rspath, save_name.lower() + '.html'))
