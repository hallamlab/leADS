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
species_name = "Salmonella"  # "Escherichia", "Salmonella"

# file
species = 'biocyc/biocyc205/mg_dataset_9257_species.pkl'
file_name = 'leADS_F'  # 'leADS_07', 'leADS_y', 'leADS_F', 'leADS_D',

# load file
with open(os.path.join(rspath, file_name + '_samples.pkl'), mode='rb') as f_in:
    samples_name = pkl.load(f_in)
with open(os.path.join(dspath, species), mode='rb') as f_in:
    species = pkl.load(f_in)
    species = species[0]
with open(os.path.join(dspath, "biocyc_y.pkl"), mode='rb') as f_in:
    y = pkl.load(f_in)
taxa = [item[1] for item in species]
species = [item[2] for item in species]
species_idx = [idx for idx, i in enumerate(species)]

samples_idx = [idx for idx, sid in enumerate(species_idx) if sid in samples_name]
samples_idx = [i for i in samples_idx if species[i].startswith(species_name)]
y = y[samples_idx]
taxa = np.array(taxa)[samples_idx]
df = pd.DataFrame(taxa)
df = df.rename(columns={0: 'Species'})
tmp = lil_matrix(y.sum(1)).toarray()
df.insert(loc=1, column="Pathway Count", value=tmp[:, 0])

alt.themes.enable('none')
chart = alt.Chart(df).properties(width=450, height=650).mark_bar(size=2, color="black").encode(
    x=alt.X('Pathway Count:Q', title="Pathway Count", sort=None),
    y=alt.Y('Species:O', title=species_name, sort=None),
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
    titleX=-40,
    grid=False
).configure_legend(
    strokeColor='gray',
    fillColor='white',
    padding=10,
    cornerRadius=10).resolve_scale(x='independent')
chart.save(os.path.join(rspath, file_name + '_' + species_name + '.html'))
