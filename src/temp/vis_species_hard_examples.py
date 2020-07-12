import altair as alt
import numpy as np
import os
import pandas as pd
import pickle as pkl
from collections import Counter
from scipy.sparse import lil_matrix

os.system('clear')

# Paths for the required files
DIRECTORY_PATH = os.getcwd()
REPO_PATH = DIRECTORY_PATH.split('/')
REPO_PATH = '/'.join(REPO_PATH[:-3])
rspath = os.path.join(REPO_PATH, 'result/leads')
dspath = os.path.join(REPO_PATH, 'dataset/biocyc/biocyc21')
ospath = os.path.join(REPO_PATH, 'objectset')

# arguments
top_pathways = 100
top_comm = 20
num_subsamples = 100
top_species = True
top_num_species = 100
num_species = 100
species_name = "Salmonella"

# file
#pathologic = 'biocyc/biocyc205/biocyc205_tier23_9255_y.pkl'
#leads = 'biocyc/biocyc205/biocyc205_tier23_9255_leads_y.pkl'
#triumpf = 'biocyc/biocyc205/biocyc205_tier23_9255_triumpf_y.pkl'

file_name = 'leADS_pp'  # 'leADS_ey' (48 samples and 496 distinct pathways), 'leADS_ml' (45 samples and 494 distinct pathways), 
                        # 'leADS_vr' (73 samples and 537 distinct pathways), 'leADS_pp' (45 samples and 505 distinct pathways) 
y_name = 'biocyc21_tier23_9429_y.pkl'
species = 'biocyc21_tier23_9429_species.pkl'

# load file
# with open(os.path.join(dspath, pathologic), mode='rb') as f_in:
# y_pathologic = pkl.load(f_in)
with open(os.path.join(rspath, file_name + '_samples.pkl'), mode='rb') as f_in:
    samples_name = pkl.load(f_in)
with open(os.path.join(dspath, species), mode='rb') as f_in:
    species = pkl.load(f_in)
    species = species[0]
with open(os.path.join(dspath, y_name), mode='rb') as f_in:
    y = pkl.load(f_in)
taxa = [item[1] for item in species]
species = [item[2] for item in species]
species_idx = [idx for idx, i in enumerate(species)]

# 1. Avergae genes to each EC according to pathway mapping
# with open(os.path.join(ospath, "biocyc.pkl"), mode='rb') as f_in:
#     data_object = pkl.load(f_in)
# pathway_dict = data_object["pathway_id"]
# enzyme_id = data_object["enzyme_id"]
# gene_ec = [(len(data_object['processed_kb']['metacyc'][5][ptwy_id][12][1]), len(data_object['processed_kb']['metacyc'][5][ptwy_id][16][1]))
#            for ptwy_id in pathway_dict.keys() if len(data_object['processed_kb']['metacyc'][5][ptwy_id][12][1]) != 0]
# del data_object
# gene_ec = np.array(gene_ec)
# gene_ec = gene_ec.sum(0)
# print(">> Avergae genes to each EC according to pathway mapping: ",
#       (gene_ec[0] + 0.0001) / (gene_ec[1] + 0.0001))

# 2. Display new pathways
# with open(os.path.join(dspath, pathologic), mode='rb') as f_in:
#     y_pathologic = pkl.load(f_in)
# with open(os.path.join(dspath, leads), mode='rb') as f_in:
#     y_leads = pkl.load(f_in)
# with open(os.path.join(dspath, triumpf), mode='rb') as f_in:
#     y_triumpf = pkl.load(f_in)

# y_leads = lil_matrix(y_leads.toarray() - y_pathologic.toarray())
# y_triumpf = lil_matrix(y_triumpf.toarray() - y_pathologic.toarray())
# y_leads[y_leads < 0] = 0
# y_leads = lil_matrix(y_leads.sum(1))
# y_triumpf[y_triumpf < 0] = 0
# y_triumpf = lil_matrix(y_triumpf.sum(1))

# df = pd.DataFrame(y_leads.todense())
# df = df.rename(columns={0: 'leADS'})
# df['triUMPF'] = y_triumpf.todense()
# df['Taxa ID'] = taxa

# Display missing pathways
# with open(os.path.join(dspath, pathologic), mode='rb') as f_in:
#     y_pathologic = pkl.load(f_in)
# with open(os.path.join(dspath, leads), mode='rb') as f_in:
#     y_leads = pkl.load(f_in)
# with open(os.path.join(dspath, triumpf), mode='rb') as f_in:
#     y_triumpf = pkl.load(f_in)
# y_leads[y_leads > 0] = 0
# y_leads[y_leads < 0] = 1
# y_leads = lil_matrix(y_leads.sum(1))
# y_triumpf[y_triumpf > 0] = 0
# y_triumpf[y_triumpf < 0] = 1
# y_triumpf = lil_matrix(y_triumpf.sum(1))

# df = pd.DataFrame(y_leads.todense())
# df = df.rename(columns={0: 'leADS'})
# df['triUMPF'] = y_triumpf.todense()
# df['Taxa ID'] = taxa
# df.index.names = ['PGDB']species

tmp = [i for i in np.arange(y.shape[0]) if species[i].startswith(species_name)]
print("## Number of pathways for {0}: {1}".format(species_name, y[tmp].sum(0).nonzero()[1].shape[0]))
samples_idx = [idx for idx, sid in enumerate(species_idx) if sid in samples_name]
tmp = [i for i in samples_idx if species[i].startswith(species_name)]
print("## Number of pathways (leADS) for {0}: {1}".format(species_name, y[tmp].sum(0).nonzero()[1].shape[0]))

if top_species:
    tmp = Counter(species).most_common(top_num_species)
    tmp = [item[0] for item in tmp]
    species_idx = [idx for idx, sp in enumerate(species) if sp in tmp]
    species = np.array(species)[species_idx]
    species = list(species)
else:
    if num_species < len(species):
        species_idx = np.random.choice(species_idx, num_species, False)
        species = np.array(species)[species_idx]
        species = list(species)

species_soft = species
species_soft = Counter(species_soft)
df = pd.DataFrame.from_dict(species_soft, orient='index').reset_index()
df = df.rename(columns={'index': 'Species', 0: 'All'})

samples_idx = [idx for idx, sid in enumerate(species_idx) if sid in samples_name]
species_hard = np.array(species)[samples_idx]
species_hard = Counter(species_hard)
tmp = pd.DataFrame.from_dict(species_hard, orient='index').reset_index()
tmp = tmp.rename(columns={'index': 'Species', 0: 'leADS'})
df = pd.merge(left=df, right=tmp, how='left',
              left_on='Species', right_on='Species')
df = df.fillna(0)
df['leADS'] = df['leADS'].astype(int)
df = df.melt(['Species'], var_name='Subsampling', value_name='Count')

alt.themes.enable('none')
chart = alt.Chart(df).properties(width=500, height=1000).mark_bar(size=10).encode(
    x=alt.X('Count:Q', title="Count", sort=None),
    y=alt.Y('Species:O', sort=None),
    color=alt.Color('Subsampling:N', scale=alt.Scale(range=['grey', 'black'])),
).configure_header(
    titleFontSize=15,
    labelFontSize=15
).configure_axis(
    labelLimit=500,
    titleFontSize=15,
    labelFontSize=10,
    labelPadding=5
).configure_axisX(
    labelFontSize=10,
    titleAngle=0,
    grid=False
).configure_axisY(
    labelFontSize=10,
    titleAngle=0,
    titleY=-10,
    titleX=-70,
    grid=False
).configure_legend(
    strokeColor='gray',
    fillColor='white',
    padding=10,
    cornerRadius=10).resolve_scale(x='independent')
chart.save(os.path.join(rspath, file_name.lower() + '_species.html'))

file_name = os.path.join(rspath, file_name.lower() + '_species.txt')
with open(file=file_name, mode='w') as fout:
    fout.write(">> Mapping species to appearance (hard examples):\n")
    for k, v in species_hard.most_common():
        fout.write("\t" + k + " & " + str(v) + "\n")
