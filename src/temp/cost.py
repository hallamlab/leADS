import numpy as np
import os
import pandas as pd
import sys
import traceback

os.system('clear')

## Paths for the required files
DIRECTORY_PATH = os.getcwd()
REPO_PATH = DIRECTORY_PATH.split('/')
REPO_PATH = '/'.join(REPO_PATH[:-3])
rspath = os.path.join(REPO_PATH, 'result/leads')


def load_data(file_name, load_path, tag='data_full', print_tag=True):
    '''
    :param data_full:
    :param load_path: load file from a path
    :type file_name: string
    :param file_name:
    '''
    try:
        if print_tag:
            print('\t\t## Loading {0:s} from: {1:s}'.format(tag, file_name))
        file_name = os.path.join(load_path, file_name)
        with open(file_name, mode='r') as f_in:
            data = f_in.readlines()
            return data
    except Exception as e:
        print('\t\t## The file {0:s} can not be loaded or located'.format(file_name), file=sys.stderr)
        print(traceback.print_exc())
        raise e


def build_dataframe(lst_cost_files, load_path):
    cost_files = sorted([f for f in lst_cost_files if str(f).endswith("cost.txt")])
    lst_costs = list()
    total_count = len(cost_files)
    print(">> Calculating costs...")
    for idx, file_name in enumerate(cost_files):
        desc = "\t--> Completed {0:.4f}%".format((idx + 1) / total_count * 100)
        if (idx + 1) != total_count:
            print(desc, end="\r")
        if (idx + 1) == total_count:
            print(desc)
        costs = load_data(file_name=file_name, load_path=load_path, print_tag=False)
        times = [float(str(d).split()[1]) for d in costs[1:]]
        costs = [float(str(d).split()[2]) for d in costs[1:]]
        lst_costs.append([file_name, np.mean(times), np.std(times), np.mean(costs), np.std(costs)])
    df = pd.DataFrame(lst_costs,
                      columns=["Mehtod", "AverageTime", "StdTime", "AverageCost", "StdCost"])
    df.to_csv(path_or_buf=os.path.join(load_path, 'cost_leads.tsv'), sep='\t')


lst_cost_files = sorted([f for f in os.listdir(rspath) if os.path.exists(os.path.join(rspath, f))])
build_dataframe(lst_cost_files=lst_cost_files, load_path=rspath)
