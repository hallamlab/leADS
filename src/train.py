'''
This file is the main entry used to train the input dataset
using leADS train and also report the predicted vocab.
'''

import numpy as np
import os
import sys
import time
import traceback
from joblib import Parallel, delayed
from model.leads import leADS
from scipy.sparse import lil_matrix, hstack
from sklearn import preprocessing
from utility.access_file import load_data, load_item_features, save_data
from utility.model_utils import score, synthesize_report, compute_abd_cov
from utility.parse_input import parse_files


def __build_features(X, pathwat_dict, ec_dict, labels_components, node2idx_pathway2ec, path2vec_features, file_name,
                     dspath, batch_size=100, num_jobs=1):
    tmp = lil_matrix.copy(X)
    print('\t>> Build abundance and coverage features...')
    list_batches = np.arange(start=0, stop=tmp.shape[0], step=batch_size)
    total_progress = len(list_batches) * len(pathwat_dict.keys())
    parallel = Parallel(n_jobs=num_jobs, verbose=0)
    results = parallel(delayed(compute_abd_cov)(tmp[batch:batch + batch_size],
                                                labels_components, pathwat_dict,
                                                None, batch_idx, total_progress)
                       for batch_idx, batch in enumerate(list_batches))
    desc = '\t\t--> Building {0:.4f}%...'.format((100))
    print(desc)
    abd, cov = zip(*results)
    abd = np.vstack(abd)
    cov = np.vstack(cov)
    del results
    abd = preprocessing.normalize(abd)
    print('\t>> Use pathway2vec EC features...')
    path2vec_features = path2vec_features[path2vec_features.files[0]]
    path2vec_features = path2vec_features / \
                        np.linalg.norm(path2vec_features, axis=1)[:, np.newaxis]
    ec_features = [idx for idx,
                           v in ec_dict.items() if v in node2idx_pathway2ec]
    path2vec_features = path2vec_features[ec_features, :]
    ec_features = [np.mean(path2vec_features[row.rows[0]] * np.array(row.data[0])[:, None], axis=0)
                   for idx, row in enumerate(X)]
    save_data(data=lil_matrix(ec_features), file_name=file_name + "_Xp.pkl", save_path=dspath, mode="wb",
              tag="transformed instances to ec features")
    X = lil_matrix(hstack((tmp, ec_features)))
    save_data(data=X, file_name=file_name + "_Xe.pkl", save_path=dspath, mode="wb",
              tag="concatenated ec features with instances")
    X = lil_matrix(hstack((tmp, abd)))
    save_data(data=X, file_name=file_name + "_Xa.pkl", save_path=dspath, mode="wb",
              tag="concatenated abundance features with instances")
    X = lil_matrix(hstack((tmp, cov)))
    save_data(data=X, file_name=file_name + "_Xc.pkl", save_path=dspath, mode="wb",
              tag="concatenated coverage features with instances")
    X = lil_matrix(hstack((tmp, ec_features)))
    X = lil_matrix(hstack((X, abd)))
    save_data(data=X, file_name=file_name + "_Xea.pkl", save_path=dspath, mode="wb",
              tag="concatenated ec and abundance features with instances")
    X = lil_matrix(hstack((tmp, ec_features)))
    X = lil_matrix(hstack((X, cov)))
    save_data(data=X, file_name=file_name + "_Xec.pkl", save_path=dspath, mode="wb",
              tag="concatenated ec and coverage features with instances")
    X = lil_matrix(hstack((tmp, ec_features)))
    X = lil_matrix(hstack((X, abd)))
    X = lil_matrix(hstack((X, cov)))
    save_data(data=X, file_name=file_name + "_Xm.pkl", save_path=dspath, mode="wb",
              tag="concatenated ec, abundance, and coverage features features with instances")


###***************************        Private Main Entry        ***************************###

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
        print('\t>> Loading files...')
        X = load_data(file_name=arg.X_name,
                      load_path=arg.dspath, tag="instances")
        X = X[:, :arg.cutting_point]

        # load a biocyc file
        data_object = load_data(file_name=arg.object_name,
                                load_path=arg.ospath, tag='the biocyc object')
        ec_dict = data_object["ec_id"]
        pathway_dict = data_object["pathway_id"]
        del data_object

        pathway_dict = dict((idx, id) for id, idx in pathway_dict.items())
        ec_dict = dict((idx, id) for id, idx in ec_dict.items())
        labels_components = load_data(
            file_name=arg.pathway2ec_name, load_path=arg.ospath, tag='M')
        print('\t>> Loading label to component mapping file object...')
        pathway2ec_idx = load_data(
            file_name=arg.pathway2ec_idx_name, load_path=arg.ospath, print_tag=False)
        pathway2ec_idx = list(pathway2ec_idx)
        tmp = list(ec_dict.keys())
        ec_dict = dict((idx, ec_dict[tmp.index(ec)])
                       for idx, ec in enumerate(pathway2ec_idx))

        # load path2vec features
        path2vec_features = np.load(
            file=os.path.join(arg.mdpath, arg.features_name))

        # load a hin file
        hin = load_data(file_name=arg.hin_name, load_path=arg.ospath,
                        tag='heterogeneous information network')
        # get pathway2ec mapping
        node2idx_pathway2ec = [node[0] for node in hin.nodes(data=True)]
        del hin

        __build_features(X=X, pathwat_dict=pathway_dict, ec_dict=ec_dict, labels_components=labels_components,
                         node2idx_pathway2ec=node2idx_pathway2ec,
                         path2vec_features=path2vec_features, file_name=arg.file_name, dspath=arg.dspath,
                         batch_size=arg.batch, num_jobs=arg.num_jobs)

    ##########################################################################################################
    ######################                       TRAIN USING leADS                       ######################
    ##########################################################################################################

    if arg.train:
        print(
            '\n{0})- Training {1} dataset using leADS model...'.format(steps, arg.X_name))
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
            y_Bags = load_data(file_name=arg.yB_name,
                               load_path=arg.dspath, tag="B")
            bags_labels = load_data(file_name=arg.bags_labels, load_path=arg.dspath,
                                    tag="bags_labels with associated pathways")
            # TODO: comment below
            # label_features = np.load(file=os.path.join(arg.dspath, arg.features_name))
            # label_features = label_features[label_features.files[0]]
            ###
            label_features = load_data(
                file_name=arg.features_name, load_path=arg.dspath, tag="features")
            centroids = np.load(file=os.path.join(arg.dspath, arg.centroids))
            centroids = centroids[centroids.files[0]]
        A = None
        if arg.fuse_weight:
            A = load_item_features(file_name=os.path.join(arg.ospath, arg.similarity_name),
                                   use_components=False)
        if arg.train_selected_sample:
            if os.path.exists(os.path.join(arg.rspath, arg.samples_ids)):
                sample_ids = load_data(
                    file_name=arg.samples_ids, load_path=arg.rspath, tag="selected samples")
                sample_ids = np.array(sample_ids)
                X = X[sample_ids, :]
                y = y[sample_ids, :]
                if not arg.train_labels:
                    y_Bags = y_Bags[sample_ids, :]
            else:
                print('\t\t No sample ids file is provided...')

        # TODO: delete below
        # from skmultilearn.dataset import load_dataset
        # X, y, _, _ = load_dataset('emotions', 'train')
        # sample = 500
        # X = X[:sample, ]
        # y = y[:sample, ]
        # y_Bags = y_Bags[:sample, ]
        ###
        model = leADS(alpha=arg.alpha, binarize_input_feature=arg.binarize_input_feature,
                      normalize_input_feature=arg.normalize_input_feature,
                      use_external_features=arg.use_external_features, cutting_point=arg.cutting_point,
                      fit_intercept=arg.fit_intercept, decision_threshold=arg.decision_threshold,
                      subsample_input_size=arg.ssample_input_size, subsample_labels_size=arg.ssample_label_size,
                      calc_ads=arg.calc_ads, acquisition_type=arg.acquisition_type, top_k=arg.top_k,
                      ads_percent=arg.ads_percent, advanced_subsampling=arg.advanced_subsampling,
                      tol_labels_iter=arg.tol_labels_iter, cost_subsample_size=arg.calc_subsample_size,
                      calc_label_cost=arg.calc_label_cost, calc_bag_cost=arg.calc_bag_cost,
                      calc_total_cost=arg.calc_total_cost, label_uncertainty_type=arg.label_uncertainty_type,
                      label_bag_sim=arg.label_bag_sim, label_closeness_sim=arg.label_closeness_sim,
                      corr_bag_sim=arg.corr_bag_sim,
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
        # X, y, _, _ = load_dataset('emotions', 'test')
        # _, y_pred = model.predict(X=X, estimate_prob=arg.estimate_prob, pred_bags=arg.pred_bags,
        #                           pred_labels=arg.pred_labels, build_up=arg.build_up, cal_average=arg.cal_average,
        #                           apply_t_criterion=arg.apply_tcriterion, adaptive_beta=arg.adaptive_beta,
        #                           decision_threshold=arg.decision_threshold, batch_size=arg.batch,
        #                           num_jobs=arg.num_jobs)
        # from sklearn.metrics import f1_score, accuracy_score, hamming_loss,confusion_matrix
        # t = f1_score(y.toarray(), y_pred, average="samples")
        # t = accuracy_score(y.toarray(), y_pred)
        # t = hamming_loss(y.toarray(), y_pred)
        # tn, fp, fn, tp = confusion_matrix(y.toarray().flatten(), y_pred.flatten()).ravel()

    ##########################################################################################################
    ######################                     EVALUATE USING leADS                      ######################
    ##########################################################################################################

    if arg.evaluate:
        print('\n{0})- Evaluating leADS model...'.format(steps))
        steps = steps + 1

        # load files
        print('\t>> Loading files...')
        X = load_data(file_name=arg.X_name, load_path=arg.dspath, tag="X")
        bags_labels = None
        label_features = None
        centroids = None
        if not arg.pred_bags:
            y = load_data(file_name=arg.y_name, load_path=arg.dspath, tag="y")
        if arg.pred_bags:
            y_Bags = load_data(file_name=arg.yB_name,
                               load_path=arg.dspath, tag="B")

        # load model
        model = load_data(file_name=arg.model_name + '.pkl',
                          load_path=arg.mdpath, tag='leADS')

        if model.learn_bags:
            bags_labels = load_data(file_name=arg.bags_labels, load_path=arg.dspath,
                                    tag="bags_labels with associated pathways")
        if model.label_uncertainty_type == "dependent":
            # TODO: comment below
            # label_features = np.load(file=os.path.join(arg.dspath, arg.features_name))
            # label_features = label_features[label_features.files[0]]
            ###
            label_features = load_data(
                file_name=arg.features_name, load_path=arg.dspath, tag="features")
            centroids = np.load(file=os.path.join(arg.dspath, arg.centroids))
            centroids = centroids[centroids.files[0]]

        # labels prediction score
        y_pred_Bags, y_pred = model.predict(X=X, bags_labels=bags_labels, label_features=label_features,
                                            centroids=centroids,
                                            estimate_prob=arg.estimate_prob, pred_bags=arg.pred_bags,
                                            pred_labels=arg.pred_labels,
                                            build_up=arg.build_up, pref_rank=arg.pref_rank, top_k_rank=arg.top_k_rank,
                                            subsample_labels_size=arg.ssample_label_size, soft_voting=arg.soft_voting,
                                            apply_t_criterion=arg.apply_tcriterion, adaptive_beta=arg.adaptive_beta,
                                            decision_threshold=arg.decision_threshold, batch_size=arg.batch,
                                            num_jobs=arg.num_jobs)

        file_name = arg.file_name + '_scores.txt'
        if arg.pred_bags:
            score(y_true=y_Bags.toarray(), y_pred=y_pred_Bags.toarray(), item_lst=['biocyc_bags'],
                  six_db=False, top_k=arg.psp_k, mode='a', file_name=file_name, save_path=arg.rspath)
        if arg.pred_labels:
            score(y_true=y.toarray(), y_pred=y_pred.toarray(), item_lst=['biocyc'], six_db=False,
                  top_k=arg.psp_k, mode='a', file_name=file_name, save_path=arg.rspath)
            if arg.dsname == 'golden':
                score(y_true=y.toarray(), y_pred=y_pred.toarray(), item_lst=['biocyc'], six_db=True,
                      top_k=arg.psp_k, mode='a', file_name=file_name, save_path=arg.rspath)

    ##########################################################################################################
    ######################                      PREDICT USING leADS                      ######################
    ##########################################################################################################

    if arg.predict:
        print(
            '\n{0})- Predicting dataset using a pre-trained leADS model...'.format(steps))
        if arg.pathway_report:
            print('\t>> Loading biocyc object...')
            # load a biocyc file
            data_object = load_data(file_name=arg.object_name, load_path=arg.ospath, tag='the biocyc object',
                                    print_tag=False)
            pathway_dict = data_object["pathway_id"]
            pathway_common_names = dict((pidx, data_object['processed_kb']['metacyc'][5][pid][0][1])
                                        for pid, pidx in pathway_dict.items()
                                        if pid in data_object['processed_kb']['metacyc'][5])
            ec_dict = data_object['ec_id']
            del data_object
            pathway_dict = dict((idx, id) for id, idx in pathway_dict.items())
            ec_dict = dict((idx, id) for id, idx in ec_dict.items())
            labels_components = load_data(
                file_name=arg.pathway2ec_name, load_path=arg.ospath, tag='M')
            print('\t>> Loading label to component mapping file object...')
            pathway2ec_idx = load_data(
                file_name=arg.pathway2ec_idx_name, load_path=arg.ospath, print_tag=False)
            pathway2ec_idx = list(pathway2ec_idx)
            tmp = list(ec_dict.keys())
            ec_dict = dict((idx, ec_dict[tmp.index(ec)])
                           for idx, ec in enumerate(pathway2ec_idx))
            if arg.extract_pf:
                X, sample_ids = parse_files(ec_dict=ec_dict, input_folder=arg.dsfolder, rsfolder=arg.rsfolder,
                                            rspath=arg.rspath, num_jobs=arg.num_jobs)
                print('\t>> Storing X and sample_ids...')
                save_data(data=X, file_name=arg.file_name + '_X.pkl', save_path=arg.dspath,
                          tag='the pf dataset (X)', mode='w+b', print_tag=False)
                save_data(data=sample_ids, file_name=arg.file_name + '_ids.pkl', save_path=arg.dspath,
                          tag='samples ids', mode='w+b', print_tag=False)
                if arg.build_features:
                    # load a hin file
                    print('\t>> Loading heterogeneous information network file...')
                    hin = load_data(file_name=arg.hin_name, load_path=arg.ospath,
                                    tag='heterogeneous information network',
                                    print_tag=False)
                    # get pathway2ec mapping
                    node2idx_pathway2ec = [node[0]
                                           for node in hin.nodes(data=True)]
                    del hin
                    print('\t>> Loading path2vec_features file...')
                    path2vec_features = np.load(
                        file=os.path.join(arg.mdpath, arg.features_name))
                    __build_features(X=X, pathwat_dict=pathway_dict, ec_dict=ec_dict,
                                     labels_components=labels_components,
                                     node2idx_pathway2ec=node2idx_pathway2ec,
                                     path2vec_features=path2vec_features,
                                     file_name=arg.file_name, dspath=arg.dspath,
                                     batch_size=arg.batch, num_jobs=arg.num_jobs)

        # load files
        print('\t>> Loading necessary files......')
        X = load_data(file_name=arg.X_name, load_path=arg.dspath, tag="X")
        sample_ids = np.arange(X.shape[0])
        if arg.samples_ids is not None:
            if arg.samples_ids in os.listdir(arg.dspath):
                sample_ids = load_data(file_name=arg.samples_ids, load_path=arg.dspath, tag="samples ids")
        tmp = lil_matrix.copy(X)
        bags_labels = None
        label_features = None
        centroids = None

        # load model
        model = load_data(file_name=arg.model_name + '.pkl',
                          load_path=arg.mdpath, tag='leADS')

        if model.learn_bags:
            bags_labels = load_data(file_name=arg.bags_labels, load_path=arg.dspath,
                                    tag="bags_labels with associated pathways")
        if model.label_uncertainty_type == "dependent":
            # TODO: comment below
            label_features = np.load(
                file=os.path.join(arg.dspath, arg.features_name))
            label_features = label_features[label_features.files[0]]
            ##
            label_features = load_data(
                file_name=arg.features_name, load_path=arg.dspath, tag="features")
            centroids = np.load(file=os.path.join(arg.dspath, arg.centroids))
            centroids = centroids[centroids.files[0]]

        # TODO: comment below
        # model.get_informative_points(X=X)
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
        if arg.pathway_report:
            print('\t>> Synthesizing pathway reports...')
            X = tmp
            synthesize_report(X=X[:, :arg.cutting_point], sample_ids=sample_ids,
                              y_pred=y_pred, y_dict_ids=pathway_dict, y_common_name=pathway_common_names,
                              component_dict=ec_dict, labels_components=labels_components, y_pred_score=y_pred_score,
                              batch_size=arg.batch, num_jobs=arg.num_jobs, rsfolder=arg.rsfolder, rspath=arg.rspath,
                              dspath=arg.dspath, file_name=arg.file_name + '_leads')
        else:
            print('\t>> Storing predictions (label index) to: {0:s}'.format(
                arg.file_name + '_leads_y.pkl'))
            save_data(data=y_pred, file_name=arg.file_name + "_leads_y.pkl", save_path=arg.dspath,
                      mode="wb", print_tag=False)
            if arg.pred_bags:
                print('\t>> Storing predictions (bag index) to: {0:s}'.format(
                    arg.file_name + '_leads_yBags.pkl'))
                save_data(data=y_pred_Bags, file_name=arg.file_name + "_leads_yBags.pkl", save_path=arg.dspath,
                          mode="wb", print_tag=False)


def train(arg):
    try:
        if arg.preprocess_dataset or arg.train or arg.evaluate or arg.predict:
            actions = list()
            if arg.preprocess_dataset:
                actions += ['PREPROCESS DATASETs']
            if arg.train:
                actions += ['TRAIN MODELs']
            if arg.evaluate:
                actions += ['EVALUATE MODELs']
            if arg.predict:
                actions += ['PREDICT RESULTS USING SPECIFIED MODELs']
            desc = [str(item[0] + 1) + '. ' + item[1]
                    for item in zip(list(range(len(actions))), actions)]
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
