__author__ = "Abdurrahman Abul-Basher"
__date__ = '02/06/2021'
__copyright__ = "Copyright 2021, The Hallam Lab"
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Abdurrahman Abul-Basher"
__email__ = "arbasher@student.ubc.ca"
__status__ = "Production"
__description__ = "This file is the main entry to perform pathway prediction and learning using leADS."

import datetime
import json
import os
import sys
import textwrap
from argparse import ArgumentParser

import utility.file_path as fph
from train import train
from utility.arguments import Arguments


def __print_header():
    if sys.platform.startswith('win'):
        os.system("cls")
    else:
        os.system("clear")
    print('# ' + '=' * 50)
    print('Author: ' + __author__)
    print('Copyright: ' + __copyright__)
    print('License: ' + __license__)
    print('Version: ' + __version__)
    print('Maintainer: ' + __maintainer__)
    print('Email: ' + __email__)
    print('Status: ' + __status__)
    print('Date: ' + datetime.datetime.strptime(__date__, "%d/%m/%Y").strftime("%d-%B-%Y"))
    print('Description: ' + textwrap.TextWrapper(width=45, subsequent_indent='\t     ').fill(__description__))
    print('# ' + '=' * 50)


def save_args(args):
    os.makedirs(args.log_dir, exist_ok=args.exist_ok)
    with open(os.path.join(args.log_dir, 'config.json'), 'wt') as f:
        json.dump(vars(args), f, indent=2)


def __internal_args(parse_args):
    arg = Arguments()

    ###***************************         Global arguments         ***************************###

    arg.display_interval = parse_args.display_interval
    if parse_args.display_interval < 0:
        arg.display_interval = 1
    arg.random_state = parse_args.random_state
    arg.num_jobs = parse_args.num_jobs
    arg.num_models = parse_args.num_models
    arg.batch = parse_args.batch
    arg.max_inner_iter = parse_args.max_inner_iter
    arg.num_epochs = parse_args.num_epochs
    arg.shuffle = parse_args.shuffle

    ###***************************          Path arguments          ***************************###

    arg.ospath = parse_args.ospath
    arg.dspath = parse_args.dspath
    arg.dsfolder = parse_args.dsfolder
    arg.mdpath = parse_args.mdpath
    arg.rspath = parse_args.rspath
    arg.logpath = parse_args.logpath

    ###***************************          File arguments          ***************************###

    arg.object_name = parse_args.object_name
    arg.pathway2ec_name = parse_args.pathway2ec_name
    arg.pathway2ec_idx_name = parse_args.pathway2ec_idx_name
    arg.features_name = parse_args.features_name
    arg.hin_name = parse_args.hin_name
    arg.X_name = parse_args.X_name
    arg.y_name = parse_args.y_name
    arg.yB_name = parse_args.yB_name
    arg.bags_labels = parse_args.bags_labels
    arg.centroids = parse_args.centroids
    arg.similarity_name = parse_args.similarity_name
    arg.file_name = parse_args.file_name
    arg.samples_ids = parse_args.samples_ids
    arg.model_name = parse_args.model_name
    arg.dsname = parse_args.dsname

    ###***************************     Preprocessing arguments      ***************************###

    arg.preprocess_dataset = parse_args.preprocess_dataset
    arg.binarize_input_feature = True
    if parse_args.no_binarize:
        arg.binarize_input_feature = False
    arg.normalize_input_feature = parse_args.normalize
    if arg.normalize_input_feature:
        arg.binarize_input_feature = False
    arg.use_external_features = True
    if parse_args.no_external_features:
        arg.use_external_features = False
    arg.cutting_point = parse_args.cutting_point

    ###***************************        Training arguments        ***************************###

    arg.train = parse_args.train
    arg.train_labels = parse_args.train_labels
    arg.fit_intercept = parse_args.fit_intercept
    arg.train_selected_sample = parse_args.train_selected_sample
    arg.ssample_input_size = parse_args.ssample_input_size
    arg.ssample_label_size = parse_args.ssample_label_size
    arg.calc_subsample_size = parse_args.calc_subsample_size
    arg.calc_label_cost = parse_args.calc_label_cost
    arg.calc_bag_cost = parse_args.calc_bag_cost
    arg.calc_total_cost = parse_args.calc_total_cost
    arg.label_bag_sim = parse_args.label_bag_sim
    arg.label_closeness_sim = parse_args.label_closeness_sim
    arg.corr_bag_sim = parse_args.corr_bag_sim
    arg.corr_label_sim = parse_args.corr_label_sim
    arg.corr_input_sim = parse_args.corr_input_sim
    arg.early_stop = parse_args.early_stop
    arg.loss_threshold = parse_args.loss_threshold

    # apply active dataset subsampling
    arg.calc_ads = parse_args.calc_ads
    arg.label_uncertainty_type = parse_args.label_uncertainty_type
    arg.ads_percent = parse_args.ads_percent
    arg.acquisition_type = parse_args.acquisition_type
    arg.top_k = parse_args.top_k
    arg.advanced_subsampling = parse_args.advanced_subsampling
    arg.tol_labels_iter = parse_args.tol_labels_iter

    # apply hyperparameters
    arg.sigma = parse_args.sigma
    arg.alpha = parse_args.alpha

    # apply regularization
    arg.penalty = parse_args.penalty
    arg.fuse_weight = parse_args.fuse_weight
    arg.alpha_elastic = parse_args.alpha_elastic
    arg.l1_ratio = 1 - parse_args.l2_ratio
    arg.lambdas = parse_args.lambdas

    # apply learning hyperparameter
    arg.learning_type = parse_args.learning_type
    arg.lr = parse_args.lr
    arg.lr0 = parse_args.lr0
    arg.forgetting_rate = parse_args.fr
    arg.delay_factor = parse_args.delay

    ###***************************       Prediction arguments       ***************************###

    arg.evaluate = parse_args.evaluate
    arg.predict = parse_args.predict
    arg.pathway_report = parse_args.pathway_report
    arg.extract_pf = False
    if parse_args.parse_pf:
        arg.extract_pf = True
    arg.pred_bags = parse_args.pred_bags
    arg.pred_labels = parse_args.pred_labels
    arg.build_up = parse_args.build_up
    arg.decision_threshold = parse_args.decision_threshold
    arg.soft_voting = parse_args.soft_voting
    arg.pref_rank = parse_args.pref_rank
    arg.top_k_rank = parse_args.top_k_rank
    arg.estimate_prob = parse_args.estimate_prob
    arg.apply_tcriterion = parse_args.apply_tcriterion
    arg.adaptive_beta = parse_args.adaptive_beta
    return arg


def parse_command_line():
    __print_header()
    # Parses the arguments.
    parser = ArgumentParser(description="Run leADS.")

    parser.add_argument('--display-interval', default=1, type=int,
                        help='display intervals. -1 means display per each iteration. (default value: 1).')
    parser.add_argument('--random-state', default=12345, type=int, help='Random seed. (default value: 12345).')
    parser.add_argument('--num-jobs', type=int, default=2, help='Number of parallel workers. (default value: 2).')
    parser.add_argument('--num-models', default=10, type=int, help='Number of models to generate. (default value: 10).')
    parser.add_argument('--batch', type=int, default=50, help='Batch size. (default value: 50).')
    parser.add_argument('--max-inner-iter', default=100, type=int,
                        help='Number of inner iteration for logistic regression. '
                             '10. (default value: 100)')
    parser.add_argument('--num-epochs', default=10, type=int,
                        help='Number of epochs over the training set. (default value: 10).')

    # Arguments for path--build-features
    parser.add_argument('--ospath', default=fph.OBJECT_PATH, type=str,
                        help='The path to the data object that contains extracted '
                             'information from the MetaCyc database. The default is '
                             'set to object folder outside the source code.')
    parser.add_argument('--dspath', default=fph.DATASET_PATH, type=str,
                        help='The path to the dataset after the samples are processed. '
                             'The default is set to dataset folder outside the source code.')
    parser.add_argument('--dsfolder', default="temp", type=str,
                        help='The dataset folder name. The default is set to temp.')
    parser.add_argument('--mdpath', default=fph.MODEL_PATH, type=str,
                        help='The path to the output models. The default is set to '
                             'train folder outside the source code.')
    parser.add_argument('--rspath', default=fph.RESULT_PATH, type=str,
                        help='The path to the results. The default is set to result '
                             'folder outside the source code.')
    parser.add_argument('--logpath', default=fph.LOG_PATH, type=str,
                        help='The path to the log directory.')

    # Arguments for file names and models
    parser.add_argument('--object-name', type=str, default='biocyc.pkl',
                        help='The biocyc file name. (default value: "biocyc.pkl")')
    parser.add_argument('--pathway2ec-name', type=str, default='pathway2ec.pkl',
                        help='The pathway2ec association matrix file name. (default value: "pathway2ec.pkl")')
    parser.add_argument('--pathway2ec-idx-name', type=str, default='pathway2ec_idx.pkl',
                        help='The pathway2ec association indices file name. (default value: "pathway2ec_idx.pkl")')
    parser.add_argument('--features-name', type=str, default='pathway2vec_embeddings.npz',
                        help='The features file name. (default value: "pathway2vec_embeddings.npz")')
    parser.add_argument('--centroids', type=str, default='biocyc_bag_centroid.npz',
                        help='The bags centroids file name. (default value: "biocyc_bag_centroid.npz")')
    parser.add_argument('--hin-name', type=str, default='hin.pkl',
                        help='The hin file name. (default value: "hin.pkl")')
    parser.add_argument('--X-name', type=str, default='temp_Xe.pkl',
                        help='The X file name. (default value: "temp_Xe.pkl")')
    parser.add_argument('--y-name', type=str, default='temp_y.pkl',
                        help='The y file name. (default value: "temp_y.pkl")')
    parser.add_argument('--yB-name', type=str, default='temp_B.pkl',
                        help='The bags file name. (default value: "temp_B.pkl")')
    parser.add_argument('--samples-ids', type=str, default="temp_ids.pkl",
                        help='The samples ids file name. (default value: "temp_ids.pkl")')
    parser.add_argument('--bags-labels', type=str, default='biocyc_bag_pathway.pkl',
                        help='The bags to labels grouping file name. (default value: "biocyc_bag_pathway.pkl")')
    parser.add_argument('--similarity-name', type=str, default='pathway_similarity_cos.pkl',
                        help='The labels similarity file name. (default value: "pathway_similarity_cos.pkl")')
    parser.add_argument('--file-name', type=str, default='temp',
                        help='The file name to save an object. (default value: "temp")')
    parser.add_argument('--model-name', type=str, default='leADS',
                        help='The file name, excluding extension, to save an object. (default value: "leADS")')
    parser.add_argument('--dsname', type=str, default='golden',
                        help='The data name used for evaluation. (default value: "golden ")')

    # Arguments for preprocessing dataset
    parser.add_argument('--preprocess-dataset', action='store_true', default=False,
                        help='Preprocess biocyc collection.  (default value: False).')

    # Arguments for training and evaluation
    parser.add_argument('--train', action='store_true', default=False,
                        help='Whether to train the leADS model. (default value: False).')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='Whether to evaluate leADS\'s performances. (default value: False).')
    parser.add_argument('--predict', action='store_true', default=False,
                        help='Whether to predict bags_labels distribution from inputs using leADS. '
                             '(default value: False).')
    parser.add_argument('--pathway-report', action='store_true', default=False,
                        help='Whether to generate a detailed report for pathways for each instance. '
                             '(default value: False).')
    parser.add_argument('--parse-pf', action='store_true', default=False,
                        help='Whether to parse Pathologic format file (pf) from a folder (default value: False).')
    parser.add_argument("--alpha", type=float, default=16,
                        help="A hyper-parameter for controlling bags centroids. (default value: 16).")
    parser.add_argument('--no-binarize', action='store_true', default=False,
                        help='Whether binarize data (set feature values to 0 or 1). (default value: False).')
    parser.add_argument('--normalize', action='store_true', default=False,
                        help='Whether to normalize data. (default value: False).')
    parser.add_argument('--no-external-features', action='store_true', default=False,
                        help='Whether to use external features that are included in data. '
                             '(default value: False).')
    parser.add_argument('--cutting-point', type=int, default=3650,
                        help='The cutting point after which binarize operation is halted in data. '
                             '(default value: 3650).')
    parser.add_argument('--fit-intercept', action='store_false', default=True,
                        help='Whether the intercept should be estimated or not. (default value: True).')
    parser.add_argument("--decision-threshold", type=float, default=0.5,
                        help="The cutoff threshold for leADS. (default value: 0.5)")
    parser.add_argument('--train-selected-sample', action='store_true', default=False,
                        help='Train based on selected sample ids. (default value: False)')
    parser.add_argument('--ssample-input-size', default=0.7, type=float,
                        help='The size of subsampled input. (default value: 0.7)')
    parser.add_argument('--ssample-label-size', default=2000, type=int,
                        help='The size of subsampled labels. (default value: 2000).')
    parser.add_argument('--calc-ads', action='store_true', default=False,
                        help='Whether to subsample dataset using active dataset subsampling (ADS). (default value: False).')
    parser.add_argument('--ads-percent', type=float, default=0.7,
                        help='Active dataset subsampling size (within [0, 1]). (default value: 0.7).')
    parser.add_argument('--tol-labels-iter', type=int, default=10,
                        help='Number of iteration to perform for subsampling labels if labels '
                             'in the samples were below expected number of labels size. '
                             '(default value: 10).')
    parser.add_argument('--advanced-subsampling', action='store_true', default=False,
                        help='Whether to apply advanced subsampling dataset based on class labels. '
                             '(default value: False).')
    parser.add_argument('--calc-subsample-size', type=int, default=1000,
                        help='Compute loss on selected samples. (default value: 1000).')
    parser.add_argument("--calc-label-cost", action='store_false', default=True,
                        help="Compute label cost, i.e., cost of labels. (default value: True).")
    parser.add_argument("--calc-bag-cost", action='store_true', default=False,
                        help="Compute bag cost, i.e., cost of bags. (default value: False).")
    parser.add_argument("--calc-total-cost", action='store_true', default=False,
                        help="Compute total cost, i.e., cost of bags plus cost of labels."
                             " (default value: False).")
    parser.add_argument('--label-uncertainty-type', default='factorize', type=str,
                        choices=['factorize', 'dependent'],
                        help='The chosen model type. (default value: "factorize")')
    parser.add_argument('--acquisition-type', default='psp', type=str,
                        choices=['entropy', 'mutual', 'variation', 'psp'],
                        help='The acquisition function for estimating the predictive uncertainty. '
                             '(default value: "psp")')
    parser.add_argument('--top-k', type=int, default=50,
                        help='Top k labels to be considered for variation ratios or psp functions. (default value: 50).')
    parser.add_argument('--label-bag-sim', action='store_true', default=False,
                        help='Whether to apply similarity constraint among labels within a bag. (default value: False).')
    parser.add_argument('--label-closeness-sim', action='store_true', default=False,
                        help='Whether to apply closeness constraint of a label to other labels of a bag. '
                             '(default value: False).')
    parser.add_argument('--corr-bag-sim', action='store_true', default=False,
                        help='Whether to apply similarity constraint among bags. (default value: False).')
    parser.add_argument('--corr-label-sim', action='store_true', default=False,
                        help='Whether to apply similarity constraint among labels. (default value: False).')
    parser.add_argument('--corr-input-sim', action='store_true', default=False,
                        help='Whether to apply similarity constraint among instances. (default value: False).')
    parser.add_argument('--penalty', default='l21', type=str, choices=['l1', 'l2', 'elasticnet', 'l21'],
                        help='The penalty (aka regularization term) to be used. (default value: "l21")')
    parser.add_argument('--alpha-elastic', default=0.0001, type=float,
                        help='Constant that multiplies the regularization term to control '
                             'the amount to regularize parameters and in our paper it is lambda. '
                             '(default value: 0.0001)')
    parser.add_argument('--l2-ratio', default=0.35, type=float,
                        help='The elastic net mixing parameter, with 0 <= l2_ratio <= 1. l2_ratio=0 '
                             'corresponds to L1 penalty, l2_ratio=1 to L2. (default value: 0.35)')
    parser.add_argument('--sigma', default=2, type=float,
                        help='Constant that scales the amount of Laplacian norm regularization '
                             'parameters. (default value: 2)')
    parser.add_argument('--fuse-weight', action='store_true', default=False,
                        help='Whether to apply fused parameters technique. (default value: False).')
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.01, 0.01, 0.01, 0.01, 0.01, 10],
                        help="Six hyper-parameters for constraints. Default is [0.01, 0.01, 0.01, 0.01, 0.01, 10].")
    parser.add_argument("--loss-threshold", type=float, default=0.05,
                        help="A hyper-parameter for deciding the cutoff threshold of the differences "
                             "of loss between two consecutive rounds. (default value: 0.05).")
    parser.add_argument("--early-stop", action='store_true', default=False,
                        help="Whether to terminate training based on relative change "
                             "between two consecutive iterations. (default value: False).")
    parser.add_argument('--learning-type', default='optimal', type=str, choices=['optimal', 'sgd'],
                        help='The learning rate schedule. (default value: "optimal")')
    parser.add_argument('--lr', default=0.0001, type=float, help='The learning rate. (default value: 0.0001).')
    parser.add_argument('--lr0', default=0.0, type=float, help='The initial learning rate. (default value: 0.0).')
    parser.add_argument('--fr', type=float, default=0.9,
                        help='Forgetting rate to control how quickly old information is forgotten. The value should '
                             'be set between (0.5, 1.0] to guarantee asymptotic convergence. (default value: 0.9).')
    parser.add_argument('--delay', type=float, default=1.,
                        help='Delay factor down weights early iterations. (default value: 1).')
    parser.add_argument('--train-labels', action='store_true', default=False,
                        help='Whether to train labels only. (default value: False).')
    parser.add_argument('--pred-bags', action='store_true', default=False,
                        help='Whether to predict bags. (default value: False).')
    parser.add_argument('--pred-labels', action='store_true', default=False,
                        help='Whether to predict labels. (default value: False).')
    parser.add_argument('--build-up', action='store_true', default=False,
                        help='Whether to predict labels based on bags or not. (default value: False).')
    parser.add_argument('--soft-voting', action='store_true', default=False,
                        help='Whether to predict labels based on the calibrated sums of the '
                             'predicted probabilities from an ensemble. (default value: False).')
    parser.add_argument('--pref-rank', action='store_true', default=False,
                        help='Whether to predict labels based on ranking strategy. (default value: False).')
    parser.add_argument('--top-k-rank', type=int, default=200,
                        help='Top k labels to be considered for predicting. Only considered when'
                             ' the prediction strategy is set to "pref-rank" option. (default value: 200).')
    parser.add_argument('--estimate-prob', action='store_true', default=False,
                        help='Whether to return prediction of labels and bags as probability '
                             'estimate or not. (default value: False).')
    parser.add_argument('--apply-tcriterion', action='store_true', default=False,
                        help='Whether to employ adaptive strategy during prediction. (default value: False).')
    parser.add_argument('--adaptive-beta', default=0.45, type=float,
                        help='The adaptive beta parameter for prediction. (default value: 0.45).')
    parser.add_argument('--shuffle', action='store_false', default=True,
                        help='Whether or not the training data should be shuffled after each epoch. '
                             '(default value: True).')

    parse_args = parser.parse_args()
    args = __internal_args(parse_args)

    train(arg=args)


if __name__ == "__main__":
    parse_command_line()
