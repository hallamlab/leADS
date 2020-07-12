import altair as alt
import os
import pandas as pd
import pickle as pkl

os.system('clear')

selected_pathways = ['ARO-PWY', 'PHESYN', 'TRPSYN-PWY', 'ARGSYNBSUB-PWY', 'VALSYN-PWY',
                     'LEUSYN-PWY', 'DAPLYSINESYN-PWY', 'HOMOSER-THRESYN-PWY',
                     'ILEUSYN-PWY', 'HISTSYN-PWY', 'HOMOSER-METSYN-PWY']
selected_enzymes = [['ARO-PWY',
                     ['2-dehydro-3-deoxyphosphoheptonate aldolase', '3-dehydroquinate synthase',
                      '3-dehydroquinate dehydratase', 'shikimate dehydrogenase', 'shikimate kinase 1',
                      '3-phosphoshikimate 1-carboxyvinyltransferase', 'chorismate synthase'],
                     ['AroG', 'AroB', 'AroD', 'AroE', 'AroK', 'AroA', 'AroC'],
                     ['2.5.1.54', '4.2.3.4', '4.2.1.10', '1.1.1.25', '2.7.1.71', '2.5.1.19', '4.2.3.5'],
                     [1, 1, 2, 2, 3, 1, 1]],
                    ['PHESYN', ['fused chorismate mutase'], ['PheA'], ['5.4.99.5'], [1]],
                    ['TRPSYN-PWY',
                     ['anthranilate synthase component I', 'anthranilate synthase component II',
                      'anthranilate phosphoribosyltransferase', 'indol-3-glycerol phosphate synthase',
                      'tryptophan synthase subunit α', 'tryptophan synthase subunit beta'],
                     ['TrpE', 'TrpG', 'TrpD', 'TrpC', 'TrpA', 'TrpB'],
                     ['4.1.3.27', '4.1.3.27', '2.4.2.18', '4.1.1.48', '4.1.2.8', '4.2.1.122'],
                     [2, 3, 1, 2, 2, 2]],
                    ['ARGSYNBSUB-PWY',
                     ['carbamoyl phosphate synthetase subunit α', 'carbamoyl phosphate synthetase subunit β',
                      'ornithine carbamoyl transferase', 'argininosuccinate synthase',
                      'argininosuccinate lyase'],
                     ['CarA', 'CarB', 'ArgF', 'ArgG', 'ArgH'],
                     ['6.3.5.5', '6.3.5.5', '2.1.3.3', '6.3.4.5', '4.3.2.1'],
                     [2, 2, 2, 1, 2]],
                    ['VALSYN-PWY',
                     ['Acetohydroxy acid synthase large subunit', 'Ketol-acid reductoisomerase',
                      'dihydroxy-acid dehydratase 2'],
                     ['IIvB', 'IIvC', 'IIvD'],
                     ['2.2.1.6', '1.1.1.383', '4.2.1.9'],
                     [1, 1, 1]],
                    ['LEUSYN-PWY',
                     ['2-isopropylmalate synthase', '3-isopropylmalate dehydratase subunit',
                      'isopropylmalate isomerase small subunit', '3-isopropylmalate isomerase'],
                     ['LeuA', 'LeuC', 'LeuD', 'LeuB'],
                     ['2.3.3.13', '4.2.1.33', '4.2.1.33', '4.2.1.33'],
                     [1, 1, 1, 1]],
                    ['DAPLYSINESYN-PWY', ['aspartate kinase III', 'aspartate semialdehyde dehydrogenase',
                                          'dihydrodipicolinate synthase', 'dihydrodipcolinate reductase',
                                          'tetrahydrodipicolinate succinylase'],
                     ['LysC', 'Asd', 'DapA', 'DapB', 'DapD'],
                     ['2.7.2.4', '1.2.1.11', '4.3.3.7', '1.17.1.8', '2.3.1.117'],
                     [1, 2, 3, 2, 1]],
                    ['HOMOSER-THRESYN-PWY',
                     ['homoserine dehydrogenase I', 'homoserine kinase', 'thronine synthase'],
                     ['ThrA', 'ThrB', 'ThrC'],
                     ['2.7.2.4', '2.7.1.39', '4.2.3.1'],
                     [1, 1, 1]],
                    ['ILEUSYN-PWY',
                     ['acetohydroxyacid synthase subunit B', 'ketol-acid reductoisomerase (NADP+)',
                      'dihydroxy-acid dehydratase'],
                     ['IIvB', 'IIvC', 'IIvD'],
                     ['2.2.1.6', '1.1.1.86', '4.2.1.9'],
                     [1, 1, 1]],
                    ['HISTSYN-PWY',
                     ['ATP phosphoribosyltransferase',
                      'phosphoribosyl-AMP cyclohydrolase',
                      'imidazole-4-carboxamide isomerase*',
                      'imidazole glycerol phosphate synthase subunit', 'imidazole glycerol phosphate synthase subunit',
                      'imidazoleglycerol-phosphate dehydratase'],
                     ['HisG', 'HisI', 'HisA', 'HisF', 'HisH', 'HisB'],
                     ['2.4.2.17', '3.5.4.19', '5.3.1.16', '4.3.2.10', '4.3.2.10', '4.2.1.19'],
                     [1, 1, 1, 1, 1, 1]],
                    ['HOMOSER-METSYN-PWY',
                     ['cobalamin-independent homocysteine'],
                     ['MetE'], ['2.1.1.14'], [1]]]
selected_topologies = ['Phenylalanine', 'Phenylalanine', 'Tryptophan', 'Arginine', 'Valine', 'Leucine',
                       'Threonine', 'Threonine', 'Isoleucine', 'Histidine', 'Methionine']
alt_names = [("imidazole-4-carboxamide isomerase",
              "1-(5-phosphoribosyl)-5-[(5-phosphoribosylamino)methylideneamino]imidazole-4-carboxamide isomerase"),
             ("imidazoleglycerol-phosphate dehydratase",
              "imidazoleglycerol-phosphate dehydratase / histidinol-phosphatase"),
             ('cobalamin-independent homocysteine', 'cobalamin-independent homocysteine transmethylase'),
             ('phosphoribosyl-AMP cyclohydrolase',
              'phosphoribosyl-AMP cyclohydrolase / phosphoribosyl-ATP pyrophosphatase'),
             ('fused chorismate mutase', 'fused chorismate mutase/prephenate dehydratase')]
## Paths for the required files
DIRECTORY_PATH = os.getcwd()
REPO_PATH = DIRECTORY_PATH.split('/')
REPO_PATH = '/'.join(REPO_PATH[:-3])
rspath = os.path.join(REPO_PATH, 'result')
ospath = os.path.join(REPO_PATH, 'objectset/')

## file name
save_name = 'symbionts_mcc_ec'

## load data
with open(os.path.join(ospath, "biocyc.pkl"), mode='rb') as f_in:
    data_object = pkl.load(f_in)
pathway_dict = data_object["pathway_id"]
pathway_common_names = dict((pidx, data_object['processed_kb']['metacyc'][5][pid][0][1])
                            for pid, pidx in pathway_dict.items()
                            if pid in data_object['processed_kb']['metacyc'][5])
del data_object

## data processing
pathway_id = list()
genes_names = list()
gene_id = list()
status = list()
for pidx, enzymes in enumerate(selected_enzymes):
    if enzymes[0] not in pathway_dict:
        continue
    pathway_name = pathway_common_names[pathway_dict[enzymes[0]]]
    tmp = [pathway_name] * len(enzymes[2])
    pathway_id.extend(tmp)
    tmp = [j + " (" + str(enzymes[2][i]) + ")" for i, j in enumerate(enzymes[1])]
    genes_names.extend(tmp)
    status.extend(enzymes[4])
df = pd.DataFrame({'PathwayFrameID': pathway_id, 'GeneNames': genes_names, 'Status': status})

# add a column "Predicted"
new_series = pd.Series([])
for i in range(len(df)):
    if int(df["Status"][i]) == 0:
        new_series[i] = "N/A"
    elif int(df["Status"][i]) == 1 or int(df["Status"][i]) == 4:
        new_series[i] = "Tremblaya"
    elif int(df["Status"][i]) == 2 or int(df["Status"][i]) == 5:
        new_series[i] = "Moranella"
    else:
        new_series[i] = "Both"
df.insert(3, "Predicted", new_series)

## plot
chart = alt.Chart(df).properties(width=500, height=1000).mark_circle(size=200, opacity=1).encode(
    x=alt.X('PathwayFrameID:O', title='MetaCyc Pathways', sort=None),
    y=alt.Y('GeneNames:O', title="Enzyme (Gene)", sort=None),
    color='Predicted:N').configure_header(
    titleFontSize=20,
    labelColor='red',
    labelFontSize=15
).configure_axis(
    titleFontSize=20,
    titlePadding=200,
    labelLimit=500,
    labelFontSize=20,
    labelPadding=5,
    grid=True).configure_legend(
    title=None,
    titleFontSize=20,
    labelFontSize=15,
    strokeColor='gray',
    fillColor='#EEEEEE',
    padding=10,
    cornerRadius=10).configure_axisY(
    titleAngle=0,
    titleY=-10,
    titleX=-150).resolve_scale(
    x='independent')

## save
chart.save(os.path.join(rspath, save_name + '.html'))
