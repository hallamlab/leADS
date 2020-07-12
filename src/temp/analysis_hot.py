import numpy as np
import os
import pandas as pd
import pickle as pkl

os.system('clear')

dataset_names = [0, 1, 2, 3, 4, 5, 6, 7, 8]

## Paths for the required files
DIRECTORY_PATH = os.getcwd()
REPO_PATH = DIRECTORY_PATH.split('/')
REPO_PATH = '/'.join(REPO_PATH[:-3])
rspath = os.path.join(REPO_PATH, 'result')
dspath = os.path.join(REPO_PATH, 'dataset')

## file name
save_name = 'leads_hots'
num_samples = 7
y = "hots_4_y.pkl"
y_biocyc = "biocyc_y.pkl"
vocab = 'vocab_biocyc.pkl'
comp_file_name = 'hots_4/leads/leADS_B2_hots/collected_pathway_report.csv'
abd_file_name = 'hots_4/leads/leADS_B2_hots/collected_pathway_abd.csv'
cov_file_name = 'hots_4/leads/leADS_B2_hots/collected_pathway_cov.csv'
col_names = {"25m": ['5', '6'], "75m": ['0', '1'],
             "110m": ['2'], "500m": ['3', '4']}
ds_columns = {"5": "25m", "0": "75m", "2": "110m", "3": "500m"}

## load data
df = pd.read_csv(os.path.join(dspath, comp_file_name), sep='\t')
df = df[['Sample_ID', 'PathwayFrameID', 'Status']]
df.index.names = ['Sample_ID']
df_abd = pd.read_csv(os.path.join(dspath, abd_file_name), sep='\t')
df_cov = pd.read_csv(os.path.join(dspath, cov_file_name), sep='\t')
df_hanson = pd.read_excel(os.path.join(os.path.join(REPO_PATH, 'dataset/hots_4/'), "hanson.xls"))
selected_pathways = list(df_hanson.iloc[1:, 0])
selected_topologies = list(df_hanson.iloc[1:, 2])
selected_metabolism = list(df_hanson.iloc[1:, 3])
with open(os.path.join(dspath, vocab), mode='rb') as f_in:
    vocab = pkl.load(f_in)
with open(os.path.join(dspath, y), mode='rb') as f_in:
    y = pkl.load(f_in)
    y = y[:num_samples]
with open(os.path.join(dspath, y_biocyc), mode='rb') as f_in:
    y_biocyc = pkl.load(f_in)

## constraint based on these pathways
y_biocyc = y_biocyc.sum(0)
y_biocyc = np.nonzero(y_biocyc)[1]
y_biocyc = [vocab[int(ptwy)] for ptwy in y_biocyc]
tmp = [selected_pathways.index(ptwy) for ptwy in selected_pathways if ptwy in y_biocyc]
selected_pathways = list(np.array(selected_pathways)[tmp])
selected_topologies = list(np.array(selected_topologies)[tmp])
selected_metabolism = list(np.array(selected_metabolism)[tmp])

## data processing
common_name = list(df_abd['PathwaysCommonName'])
df_comp = pd.DataFrame(data=np.zeros((len(vocab.keys()), num_samples)), index=list(vocab.values()),
                       columns=tuple(np.arange(num_samples)))
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
df_comp.columns = [str(col) for col in df_comp.columns]
df_abd = df_abd.iloc[:, dataset_names]
df_cov = df_cov.iloc[:, dataset_names]
df_comp = df_comp.iloc[:, dataset_names]
if len(selected_pathways) == 0:
    selected_pathways = [item for idx, item in enumerate(df_comp.iloc[:, 0])]
    selected_topologies = len(selected_pathways) * [" "]
    selected_metabolism = len(selected_metabolism) * [" "]
tmp = [selected_pathways.index(item) for idx, item in enumerate(df_comp.iloc[:, 0]) if item in selected_pathways]
selected_pathways = np.array(selected_pathways)[tmp]
selected_topologies = np.array(selected_topologies)[tmp]
selected_metabolism = np.array(selected_metabolism)[tmp]
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

## add a column "Class1"
new_series = pd.Series([])
new_series_1 = pd.Series([])
for i in range(len(df_comp)):
    new_series[i] = selected_topologies[list(selected_pathways).index(df_comp.iloc[i, 0])]
    new_series_1[i] = selected_metabolism[list(selected_pathways).index(df_comp.iloc[i, 0])]
df_abd.insert(2, "Class1", new_series)
df_cov.insert(2, "Class1", new_series)
df_comp.insert(2, "Class1", new_series)
df_abd.insert(3, "Class2", new_series_1)
df_cov.insert(3, "Class2", new_series_1)
df_comp.insert(3, "Class2", new_series_1)

# drop "Pathways_ID"
df_abd.drop("PathwayFrameID", axis=1, inplace=True)
df_cov.drop("PathwayFrameID", axis=1, inplace=True)
df_comp.drop("PathwayFrameID", axis=1, inplace=True)
for k, v in col_names.items():
    header = v[0]
    v = v[1:]
    if len(v) > 0:
        for item in v:
            for i in range(len(df_comp)):
                if int(df_comp[header][i]) < int(df_comp[item][i]):
                    df_comp[header][i] = int(df_comp[item][i])
                df_abd[header][i] = df_abd[header][i] + df_abd[item][i]

for k, v in col_names.items():
    header = v[0]
    v = v[1:]
    if len(v) > 0:
        for item in v:
            df_abd.drop(item, axis=1, inplace=True)
            df_cov.drop(item, axis=1, inplace=True)
            df_comp.drop(item, axis=1, inplace=True)

# remove non-coverage pathways
remove_idx = [i for i in range(len(df_cov)) if np.sum(df_cov.loc[i][3:]) == 0]
df_abd.drop(remove_idx, inplace=True)
df_cov.drop(remove_idx, inplace=True)
df_comp.drop(remove_idx, inplace=True)

# rename ds_columns
df_abd.rename(columns=dict((k, v) for k, v in ds_columns.items()), inplace=True)
df_cov.rename(columns=dict((k, v) for k, v in ds_columns.items()), inplace=True)
df_comp.rename(columns=dict((k, v) for k, v in ds_columns.items()), inplace=True)

# melt and merge
df_abd = df_abd[['PathwaysCommonName', 'Class1', 'Class2'] + list(col_names.keys())]
df_abd = df_abd.melt(['PathwaysCommonName', 'Class1', 'Class2'], var_name='AbundanceDataSet', value_name='Abundance')
df_comp = df_comp[['PathwaysCommonName', 'Class1', 'Class2'] + list(col_names.keys())]
df_comp = df_comp.melt(['PathwaysCommonName', 'Class1', 'Class2'], var_name='DataSet', value_name='Status')
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
print(
    "Unique pathways: {0}, None: {1}, PathoLogic: {2}, ML: {3}, Both: {4}".format(len(selected_pathways), no_one, patho,
                                                                                  ml, both))
df_comp.insert(5, "Predicted", new_series)
