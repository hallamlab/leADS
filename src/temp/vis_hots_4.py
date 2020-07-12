import os
import pickle as pkl

import altair as alt
import numpy as np
import pandas as pd

os.system('clear')

selected_pathways = ['PWY-101', 'PWY-6785', 'PWY-5082', 'PWY1F-FLAVSYN', 'PWY0-1415', 'PWY-6893',
                     'PWY-6894', 'P381-PWY', 'PWY-5507', 'PWY-6908', 'PWY0-501', 'GLUTATHIONESYN-PWY',
                     'PWY0-1507', 'PWY-5123', 'PWY-5514', 'GLYCOGENSYNTH-PWY', 'PWY-801', 'PWY-6389',
                     'RIBITOLUTIL-PWY', 'PWY-4101', 'AMMOXID-PWY', 'PWY-6523', 'PWY-5674',
                     'PWY-6608', 'PWY-6713', 'MANNCAT-PWY', 'PWY-5747', 'PWY-5535', 'PWY-6038', 'PWY-5493',
                     'PWY-1641', 'THREONINE-DEG2-PWY', 'THRDLCTCAT-PWY', 'PWY-6802', 'PWY-6098', 'PWY-6654',
                     'MENAQUINONESYN-PWY', 'PWY-7729', 'PWY-5523', 'PWY1G-0', 'P261-PWY', 'PWY-6466',
                     'PWY-7865', 'PWY-5207', 'PWY-7866', 'PWY-7867', 'PWY-7868', 'PWY-6138', 'PWY-6139',
                     'PWY0-1241', 'PWY-6281', 'GLYSYN-THR-PWY']
selected_topologies = ['Energy', 'Degradation', 'Degradation', 'Biosynthesis', 'Biosynthesis',
                       'Biosynthesis', 'Biosynthesis', 'Biosynthesis', 'Biosynthesis', 'Biosynthesis',
                       'Biosynthesis', 'Biosynthesis', 'Biosynthesis', 'Biosynthesis', 'Biosynthesis',
                       'Biosynthesis', 'Biosynthesis', 'Energy', 'Degradation', 'Degradation', 'Degradation',
                       'Degradation', 'Degradation', 'Degradation', 'Degradation', 'Degradation',
                       'Degradation', 'Degradation', 'Degradation', 'Degradation', 'Degradation',
                       'Degradation', 'Degradation',
                       'Biosynthesis', 'Biosynthesis', 'Biosynthesis', 'Biosynthesis', 'Biosynthesis',
                       'Biosynthesis', 'Biosynthesis', 'Biosynthesis', 'Biosynthesis', 'Biosynthesis',
                       'Biosynthesis', 'Biosynthesis', 'Biosynthesis', 'Biosynthesis', 'Biosynthesis',
                       'Biosynthesis', 'Biosynthesis', 'Biosynthesis', 'Biosynthesis']
selected_metabolism = ['Photosynthesis', 'Hydrogen production amino acids', 'Hydrogen production amino acids',
                       'Secondary metabolites', 'Cofactors', 'Cofactors', 'Cofactors', 'Cofactors', 'Cofactors',
                       'Cofactors', 'Cofactors', 'Cofactors', 'Cofactors', 'Cofactors', 'Cofactors', 'Carbohydrates',
                       'Amino acids', 'Fermentation', 'Secondary metabolites', 'Secondary metabolites',
                       'Non-carbon nutrients', 'Non-carbon nutrients', 'Non-carbon nutrients', 'Nucleotides',
                       'Carbohydrates', 'Carbohydrates', 'Carboxylates', 'Carboxylates', 'Carboxylates',
                       'C1 compounds', 'C1 compounds', 'Amino acids', 'Amino acids', 'Secondary metabolites',
                       'Secondary metabolites', 'Cofactors', 'Cofactors', 'Cofactors', 'Cofactors', 'Cofactors',
                       'Cofactors', 'Cofactors', 'Cofactors', 'Cofactors', 'Cofactors', 'Cofactors', 'Cofactors',
                       'Carbohydrates', 'Carbohydrates', 'Carbohydrates', 'Amino acids', 'Amino acids']
dataset_names = [0, 1, 2, 3, 4, 5, 6, 7, 8]

# Paths for the required files
DIRECTORY_PATH = os.getcwd()
REPO_PATH = DIRECTORY_PATH.split('/')
REPO_PATH = '/'.join(REPO_PATH[:-3])
rspath = os.path.join(REPO_PATH, 'result')
dspath = os.path.join(REPO_PATH, 'dataset')
mllr_dspath = os.path.join(dspath, 'hots_4/mllr')
ospath = os.path.join(REPO_PATH, 'objectset')

# File name
sample_idx = 3
save_name = 'leads'
folders_name = "hots_4"
abd_file_name = 'mg_hots_4_abd.csv'
folders = ['pathologic', 'mllr', 'triumpf']
ds_names = ['PathoLogic', 'mlLGPR', 'triUMPF']
files = ["hots_4_pathologic_y.pkl", "hots_4_mllr_y.pkl", "triUMPF_e_hots_y.pkl"]
leads_names = ['Random', 'Full', 'Entropy', 'Mutual information', 'Variation ratios', 'nPSP']
leads_files = ["leADS_ra_hots_y.pkl", "leADS_fl_hots_y.pkl",
               "leADS_ey_hots_y.pkl", "leADS_ml_hots_y.pkl", 
              "leADS_vr_hots_y.pkl", "leADS_pp_hots_y.pkl"]
ds_columns = {"25m": ['srr020493cyc', 'srr020494cyc'], "75m": ['srr020488cyc', 'srr020489cyc'],
              "110m": ['srr020490cyc'], "500m": ['srr020491cyc', 'srr020492cyc']}
col_idx = {"25m": [5, 6], "75m": [0, 1], "110m": [2], "500m": [3, 4]}
tmp = list(ds_columns.keys())[sample_idx]
ds_columns = {tmp: ds_columns[tmp]}
col_idx = col_idx[tmp]
save_name = save_name + "_" + list(ds_columns.keys())[0]

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
selected_metabolism = np.array([selected_metabolism[selected_pathways.index(
    ptwy)] for ptwy in pathway_dict if ptwy in selected_pathways])
del data_object
abd = pd.read_csv(os.path.join(mllr_dspath, abd_file_name), sep='\t')
abd = abd.iloc[:, dataset_names]
abd = abd.loc[:, ['Pathways_ID', 'PathwaysCommonName'] +
              list(ds_columns.values())[0]]

# Data processing
tmp = [abd[abd['Pathways_ID'] == item].index.values.astype(
    int)[0] for item in pathway_dict]
abd = abd.iloc[tmp, :]
abd = np.sum(abd.iloc[:, 2:], axis=1)

# all models
df_comp = pd.DataFrame()
predicted = list()
for idx, f in enumerate(files):
    f = os.path.join(dspath, folders_name, folders[idx], f)
    with open(f, mode='rb') as f_in:
        tmp = pkl.load(f_in)
        tmp = tmp[:, pathway_idx]
        tmp = np.array(tmp[col_idx].sum(0).tolist()[0])
        tmp[tmp > 1] = 1
        print(tmp.sum())
        predicted.append(
            ["Yes" if item == 1 else "No" for item in list(tmp)])
        df_comp.insert(loc=idx, column=ds_names[idx], value=abd)
predicted = np.concatenate(predicted)
df_comp.insert(loc=0, column="PathwaysCommonName", value=pathway_common_names)
df_comp.insert(loc=1, column="Metabolism", value=selected_metabolism)
df_comp.insert(loc=2, column="Topology", value=selected_topologies)
df_comp = df_comp.reset_index(drop=True)

for i in range(len(df_comp)):
    df_comp["PathwaysCommonName"][i] = df_comp["PathwaysCommonName"][i] + \
        " [" + df_comp["Metabolism"][i] + "]"

df_comp = df_comp.melt(['PathwaysCommonName', 'Metabolism', 'Topology'],
                       var_name='DataSet', value_name='Abundance')
df_comp.insert(df_comp.shape[1], "Predicted", predicted)
df_comp.sort_values(by=['Topology', 'Metabolism'], inplace=True)
df_comp.drop("Metabolism", axis=1, inplace=True)

# mlts
df_leads = pd.DataFrame()
predicted = list()
for idx, f in enumerate(leads_files):
    f = os.path.join(dspath, folders_name, 'leads', f)
    with open(f, mode='rb') as f_in:
        tmp = pkl.load(f_in)
        tmp = tmp[:, pathway_idx]
        tmp = np.array(tmp[col_idx].sum(0).tolist()[0])
        tmp[tmp > 1] = 1
        print(tmp.sum())
        predicted.append(
            ["Yes" if item == 1 else "No" for item in list(tmp)])
        df_leads.insert(loc=idx, column=leads_names[idx], value=abd)
predicted = np.concatenate(predicted)
df_leads.insert(loc=0, column="PathwaysCommonName", value=pathway_common_names)
df_leads.insert(loc=1, column="Metabolism", value=selected_metabolism)
df_leads.insert(loc=2, column="Topology", value=selected_topologies)
df_leads = df_leads.reset_index(drop=True)

for i in range(len(df_leads)):
    df_leads["PathwaysCommonName"][i] = df_leads["PathwaysCommonName"][i] + \
        " [" + df_leads["Metabolism"][i] + "]"

df_leads = df_leads.melt(['PathwaysCommonName', 'Metabolism', 'Topology'],
                       var_name='DataSet', value_name='Abundance')
df_leads.insert(df_leads.shape[1], "Predicted", predicted)
df_leads.sort_values(by=['Topology', 'Metabolism'], inplace=True)
df_leads.drop("Metabolism", axis=1, inplace=True)

# Plot
chart1 = alt.Chart(df_comp).properties(width=150, height=1100).mark_circle(opacity=1).encode(
    x=alt.X('DataSet:O', title=None, sort=ds_names),
    y=alt.Y('PathwaysCommonName:O', title='MetaCyc Pathways', sort=None),
    color=alt.Color('Predicted:N', scale=alt.Scale(range=['grey', 'black'])),
    size=alt.Size('Abundance', scale=alt.Scale(domain=[-10, 100])))

chart2 = alt.Chart(df_leads).properties(width=250, height=1100).mark_circle(opacity=1).encode(
    x=alt.X('DataSet:O', title=None, sort=leads_names),
    y=alt.Y('PathwaysCommonName:O', title=None, sort=None, axis=None),
    color=alt.Color('Predicted:N', scale=alt.Scale(range=['grey', 'black'])),
    size=alt.Size('Abundance', scale=alt.Scale(domain=[-10, 100])))

chart = (chart1 | chart2).configure_header(
    titleFontSize=20,
    labelColor='red',
    labelFontSize=15
).configure_axis(
    titleFontSize=20,
    labelLimit=1500,
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
