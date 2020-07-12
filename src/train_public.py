'''
This file is the main entry used to train the input dataset
using leADS train and also report the predicted vocab.
'''

import numpy as np
import os
import sys
import time
import traceback
from model.leads import leADS
from scipy.sparse import lil_matrix, hstack
from utility.access_file import load_data, load_item_features, save_data
from utility.model_utils import synthesize_report
from utility.parse_input import parse_files


###***************************        Add Features        ***************************###

def __preprocess_dataset(X_name, ec_dict, pathway2ec_idx, cutting_point, features_name, hin_name, file_name, dspath,
                         ospath, mdpath):
    print('\t>> Loading files...')
    X = load_data(file_name=X_name, load_path=dspath, tag="instances")
    X = X[:, :cutting_point]

    # load a hin file
    hin = load_data(file_name=hin_name, load_path=ospath, tag='heterogeneous information network')
    # get pathway2ec mapping
    node2idx_pathway2ec = [node[0] for node in hin.nodes(data=True)]
    del hin
    # load path2vec features
    path2vec_features = np.load(file=os.path.join(mdpath, features_name))

    # extracting ec and pathway features
    path2vec_features = path2vec_features[path2vec_features.files[0]]
    path2vec_features = path2vec_features / np.linalg.norm(path2vec_features, axis=1)[:, np.newaxis]
    ec_features = [idx for v, idx in ec_dict.items() if v in node2idx_pathway2ec]
    ec_idx = np.array([idx for idx in ec_features if idx in pathway2ec_idx])
    path2vec_features = path2vec_features[ec_idx, :]
    print('\t>> Adopting ec features...')
    ec_features = [(np.mean(path2vec_features[row.rows[0]] * np.array(row.data[0])[:, None], axis=0))
                   for idx, row in enumerate(X)]
    save_data(data=lil_matrix(ec_features), file_name=file_name + "_Xp.pkl", save_path=dspath, mode="wb",
              tag="transformed instances to ec features")
    X = lil_matrix(hstack((X, ec_features)))
    save_data(data=X, file_name=file_name + "_Xe.pkl", save_path=dspath, mode="wb",
              tag="concatenated ec features with instances")


def __train(arg):
    # Setup the number of operations to employ
    steps = 1
    # Whether to display parameters at every operation
    display_params = True

    ##########################################################################################################
    ######################                  PREPROCESSING DATASET                       ######################
    ##########################################################################################################

    if arg.preprocess_dataset:
        print('\n{0})- Preprocess dataset...'.format(steps))
        steps = steps + 1

        # load a biocyc file
        data_object = load_data(file_name=arg.object_name, load_path=arg.ospath, tag='the biocyc object')
        ec_dict = data_object["ec_id"]
        del data_object
        pathway2ec_idx = load_data(file_name=arg.pathway2ec_idx_name, load_path=arg.ospath)
        pathway2ec_idx = list(pathway2ec_idx)

        __preprocess_dataset(X_name=arg.X_name, ec_dict=ec_dict, pathway2ec_idx=pathway2ec_idx,
                             cutting_point=arg.cutting_point, features_name=arg.features_name, hin_name=arg.hin_name,
                             file_name=arg.file_name, dspath=arg.dspath, ospath=arg.ospath, mdpath=arg.mdpath)

    ##########################################################################################################
    ######################                       TRAIN USING leADS                       #####################
    ##########################################################################################################

    if arg.train:
        print('\n{0})- Training {1} dataset using leADS model...'.format(steps, arg.X_name))
        steps = steps + 1

        # load files
        print('\t>> Loading files...')
        X = load_data(file_name=arg.X_name, load_path=arg.dspath, tag="X")
        y = load_data(file_name=arg.y_name, load_path=arg.dspath, tag="y")
        y_Bags = None
        bags_labels = None
        label_features = None
        centroids = None
        if not arg.train_labels:
            y_Bags = load_data(file_name=arg.yB_name, load_path=arg.dspath, tag="B")
            bags_labels = load_data(file_name=arg.bags_labels, load_path=arg.dspath,
                                    tag="bags_labels with associated pathways")
            label_features = load_data(file_name=arg.features_name, load_path=arg.dspath, tag="features")
            centroids = np.load(file=os.path.join(arg.dspath, arg.centroids))
            centroids = centroids[centroids.files[0]]
        A = None
        if arg.fuse_weight:
            A = load_item_features(file_name=os.path.join(arg.ospath, arg.similarity_name),
                                   use_components=False)
        if arg.train_selected_sample:
            if os.path.exists(os.path.join(arg.rspath, arg.samples_ids)):
                sample_ids = load_data(file_name=arg.samples_ids, load_path=arg.rspath, tag="selected samples")
                sample_ids = np.array(sample_ids)
                X = X[sample_ids, :]
                y = y[sample_ids, :]
                if not arg.train_labels:
                    y_Bags = y_Bags[sample_ids, :]
            else:
                print('\t\t No sample ids file is provided...')

        model = leADS(alpha=arg.alpha, binarize_input_feature=arg.binarize_input_feature,
                      normalize_input_feature=arg.normalize_input_feature,
                      use_external_features=arg.use_external_features, cutting_point=arg.cutting_point,
                      fit_intercept=arg.fit_intercept, decision_threshold=arg.decision_threshold,
                      subsample_input_size=arg.ssample_input_size, subsample_labels_size=arg.ssample_label_size,
                      calc_ads=arg.calc_ads, ads_percent=arg.ads_percent, cost_subsample_size=arg.calc_subsample_size,
                      calc_label_cost=arg.calc_label_cost, calc_bag_cost=arg.calc_bag_cost,
                      calc_total_cost=arg.calc_total_cost, label_uncertainty_type=arg.label_uncertainty_type,
                      acquisition_type=arg.acquisition_type, top_k=arg.top_k, label_bag_sim=arg.label_bag_sim,
                      label_closeness_sim=arg.label_closeness_sim, corr_bag_sim=arg.corr_bag_sim,
                      corr_label_sim=arg.corr_label_sim, corr_input_sim=arg.corr_input_sim, penalty=arg.penalty,
                      alpha_elastic=arg.alpha_elastic, l1_ratio=arg.l1_ratio, sigma=arg.sigma,
                      fuse_weight=arg.fuse_weight, lambdas=arg.lambdas, loss_threshold=arg.loss_threshold,
                      early_stop=arg.early_stop, learning_type=arg.learning_type, lr=arg.lr, lr0=arg.lr0,
                      delay_factor=arg.delay_factor, forgetting_rate=arg.forgetting_rate, num_models=arg.num_models,
                      batch=arg.batch, max_inner_iter=arg.max_inner_iter, num_epochs=arg.num_epochs,
                      num_jobs=arg.num_jobs, display_interval=arg.display_interval, shuffle=arg.shuffle,
                      random_state=arg.random_state, log_path=arg.logpath)
        model.fit(X=X, y=y, y_Bag=y_Bags, bags_labels=bags_labels, label_features=label_features, centroids=centroids,
                  A=A, model_name=arg.model_name, model_path=arg.mdpath, result_path=arg.rspath,
                  display_params=display_params)

    ##########################################################################################################
    ######################                      PREDICT USING leADS                      ######################
    ##########################################################################################################

    if arg.predict:
        print('\n{0})- Predicting dataset using a pre-trained leADS model...'.format(steps))
        print('\t>> Loading files...')
        # load a biocyc file
        data_object = load_data(file_name=arg.object_name, load_path=arg.ospath, tag='the biocyc object')
        pathway_dict = data_object["pathway_id"]
        pathway_common_names = dict((pidx, data_object['processed_kb']['metacyc'][5][pid][0][1])
                                    for pid, pidx in pathway_dict.items()
                                    if pid in data_object['processed_kb']['metacyc'][5])
        ec_dict = data_object['ec_id']
        del data_object
        pathway_dict = dict((idx, id) for id, idx in pathway_dict.items())

        if arg.extract_pf:
            pathway2ec_idx = load_data(file_name=arg.pathway2ec_idx_name, load_path=arg.ospath)
            pathway2ec_idx = list(pathway2ec_idx)
            X, sample_ids = parse_files(pathway2ec_idx=pathway2ec_idx, ec_dict=ec_dict, dsfolder=arg.dsfolder,
                                        rsfolder=arg.rsfolder, dspath=arg.dspath, rspath=arg.rspath,
                                        num_jobs=arg.num_jobs)
            save_data(data=lil_matrix(X), file_name=arg.file_name + '_X.pkl', save_path=arg.dspath,
                      tag='the pf dataset (X)', mode='w+b')
            save_data(data=sample_ids, file_name=arg.file_name + '_ids.pkl', save_path=arg.dspath,
                      tag='samples ids', mode='w+b')
            __preprocess_dataset(X_name=arg.file_name + '_X.pkl', ec_dict=ec_dict, pathway2ec_idx=pathway2ec_idx,
                                 cutting_point=arg.cutting_point, features_name=arg.features_name,
                                 hin_name=arg.hin_name, file_name=arg.file_name, dspath=arg.dspath,
                                 ospath=arg.ospath, mdpath=arg.mdpath)
            arg.X_name = arg.file_name + '_Xe.pkl'
            arg.samples_ids = arg.file_name + '_ids.pkl'

        # load files
        X = load_data(file_name=arg.X_name, load_path=arg.dspath, tag="X")
        sample_ids = np.arange(X.shape[0])
        if arg.samples_ids in os.listdir(arg.dspath):
            sample_ids = load_data(file_name=arg.samples_ids, load_path=arg.dspath, tag="samples ids")
        tmp = lil_matrix.copy(X)
        M = load_data(file_name=arg.pathway2ec_name, load_path=arg.ospath, tag='M')
        bags_labels = None
        label_features = None
        centroids = None

        # load model
        model = load_data(file_name=arg.model_name + '.pkl', load_path=arg.mdpath, tag='leADS')

        if model.learn_bags:
            bags_labels = load_data(file_name=arg.bags_labels, load_path=arg.dspath,
                                    tag="bags_labels with associated pathways")
        if model.label_uncertainty_type == "dependent":
            label_features = load_data(file_name=arg.features_name, load_path=arg.dspath, tag="features")
            centroids = np.load(file=os.path.join(arg.dspath, arg.centroids))
            centroids = centroids[centroids.files[0]]

        # predict
        y_pred_Bags, y_pred = model.predict(X=X, bags_labels=bags_labels, label_features=label_features,
                                            centroids=centroids,
                                            estimate_prob=False, pred_bags=arg.pred_bags, pred_labels=arg.pred_labels,
                                            build_up=arg.build_up, pref_rank=arg.pref_rank, top_k_rank=arg.top_k_rank,
                                            subsample_labels_size=arg.ssample_label_size, soft_voting=arg.soft_voting,
                                            apply_t_criterion=arg.apply_tcriterion, adaptive_beta=arg.adaptive_beta,
                                            decision_threshold=arg.decision_threshold, batch_size=arg.batch,
                                            num_jobs=arg.num_jobs)
        # labels prediction score
        y_pred_Bags_score, y_pred_score = model.predict(X=X, bags_labels=bags_labels, label_features=label_features,
                                                        centroids=centroids, estimate_prob=True,
                                                        pred_bags=arg.pred_bags,
                                                        pred_labels=arg.pred_labels, build_up=arg.build_up,
                                                        pref_rank=arg.pref_rank, top_k_rank=arg.top_k_rank,
                                                        subsample_labels_size=arg.ssample_label_size,
                                                        soft_voting=arg.soft_voting,
                                                        apply_t_criterion=arg.apply_tcriterion,
                                                        adaptive_beta=arg.adaptive_beta,
                                                        decision_threshold=arg.decision_threshold,
                                                        batch_size=arg.batch, num_jobs=arg.num_jobs)
        print('\t>> Synthesizing reports...')
        X = tmp
        synthesize_report(X=X[:, :arg.cutting_point], sample_ids=sample_ids,
                          y_pred=y_pred, y_dict_ids=pathway_dict, y_common_name=pathway_common_names,
                          labels_components=M, y_pred_score=y_pred_score, batch_size=arg.batch, num_jobs=arg.num_jobs,
                          rsfolder=arg.rsfolder, rspath=arg.rspath, dspath=arg.dspath, file_name=arg.file_name)


def train(arg):
    try:
        if arg.preprocess_dataset or arg.train or arg.predict:
            actions = list()
            if arg.preprocess_dataset:
                actions += ['PREPROCESS DATASETs']
            if arg.train:
                actions += ['TRAIN MODELs']
            if arg.predict:
                actions += ['PREDICT RESULTS USING SPECIFIED MODELs']
            desc = [str(item[0] + 1) + '. ' + item[1] for item in zip(list(range(len(actions))), actions)]
            desc = ' '.join(desc)
            print('\n*** APPLIED ACTIONS ARE: {0}'.format(desc))
            timeref = time.time()
            __train(arg)
            print('\n*** The selected actions consumed {1:f} SECONDS\n'.format('', round(time.time() - timeref, 3)),
                  file=sys.stderr)
        else:
            print('\n*** PLEASE SPECIFY AN ACTION...\n', file=sys.stderr)
    except Exception:
        print(traceback.print_exc())
        raise
