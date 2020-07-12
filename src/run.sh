# !/bin/bash
# set -e
# set -x

#pip install -r requirements.txt

# Note -- these are not the recommended settings for this dataset.  This is just so the open-source tests will finish quickly.

############## Preprecess the dataset
# python main.py --preprocess-dataset --object-name "biocyc.pkl" --pathway2ec-idx-name "pathway2ec_idx.pkl" --hin-name "hin_cmt.pkl" --features-name "path2vec_cmt_tf_embeddings.npz" --X-name "golden_X.pkl" --file-name "golden"
# python main.py --preprocess-dataset --object-name "biocyc.pkl" --pathway2ec-idx-name "pathway2ec_idx.pkl" --hin-name "hin_cmt.pkl" --features-name "path2vec_cmt_tf_embeddings.npz" --X-name "symbionts_X.pkl" --file-name "symbionts"
# python main.py --preprocess-dataset --object-name "biocyc.pkl" --pathway2ec-idx-name "pathway2ec_idx.pkl" --hin-name "hin_cmt.pkl" --features-name "path2vec_cmt_tf_embeddings.npz" --X-name "hots_4_X.pkl" --file-name "hots_4"
# python main.py --preprocess-dataset --object-name "biocyc.pkl" --pathway2ec-idx-name "pathway2ec_idx.pkl" --hin-name "hin_cmt.pkl" --features-name "path2vec_cmt_tf_embeddings.npz" --X-name "cami_X.pkl" --file-name "cami"
# python main.py --preprocess-dataset --object-name "biocyc.pkl" --pathway2ec-idx-name "pathway2ec_idx.pkl" --hin-name "hin_cmt.pkl" --features-name "path2vec_cmt_tf_embeddings.npz" --X-name "gutcyc_X.pkl" --file-name "gutcyc"
# python main.py --preprocess-dataset --object-name "biocyc.pkl" --pathway2ec-idx-name "pathway2ec_idx.pkl" --hin-name "hin_cmt.pkl" --features-name "path2vec_cmt_tf_embeddings.npz" --X-name "biocyc_X.pkl" --file-name "biocyc"
# python main.py --preprocess-dataset --object-name "biocyc.pkl" --pathway2ec-idx-name "pathway2ec_idx.pkl" --hin-name "hin_cmt.pkl" --features-name "path2vec_cmt_tf_embeddings.npz" --X-name "biocyc21_X.pkl" --file-name "biocyc21"
# python main.py --preprocess-dataset --object-name "biocyc.pkl" --pathway2ec-idx-name "pathway2ec_idx.pkl" --hin-name "hin_cmt.pkl" --features-name "path2vec_cmt_tf_embeddings.npz" --X-name "biocyc205_tier23_9255_X.pkl" --file-name "biocyc205_tier23"

#######################################################################################################
###################################           Optimum test          ###################################
#######################################################################################################
### Optimum results for experiments
### Shamwow
# python main.py --train --train-labels --binarize --use-external-features --calc-ads --ads-percent 0.7 --acquisition-type "entropy" --ssample-input-size 1.0 --ssample-label-size 2000 --calc-subsample-size 1000 --lambdas 0.01 0.01 0.01 0.01 0.01 10 --penalty "l21" --X-name "biocyc21_Xe.pkl" --y-name "biocyc21_y.pkl" --model-name "leADS_ey" --batch 50 --num-models 10 --num-epochs 10 --num-jobs 15 --display-interval 1
python main.py --evaluate --binarize --pred-labels --soft-voting --decision-threshold 0.5 --X-name "golden_Xe.pkl" --y-name "golden_y.pkl" --dsname "golden" --file-name "leADS_ey_1" --model-name "leADS_ey" --num-jobs 15
python main.py --evaluate --binarize --pred-labels --soft-voting --decision-threshold 0.5 --X-name "cami_Xe.pkl" --y-name "cami_y.pkl" --dsname "cami" --file-name "leADS_ey_2" --model-name "leADS_ey" --num-jobs 15
python main.py --evaluate --binarize --pred-labels --soft-voting --decision-threshold 0.5 --X-name "biocyc21_tier3_9392_Xe.pkl" --y-name "biocyc21_tier3_9392_y.pkl" --dsname "biocyc" --file-name "leADS_ey_3" --model-name "leADS_ey" --num-jobs 15
# python main.py --predict --binarize --pathway-report --no-parse --no-build-features --pred-labels --soft-voting --decision-threshold 0.5 --object-name "biocyc.pkl" --pathway2ec-idx-name "pathway2ec_idx.pkl" --pathway2ec-name 'pathway2ec.pkl' --hin-name 'hin_cmt.pkl' --features-name 'path2vec_cmt_tf_embeddings.npz' --X-name 'symbionts_Xe.pkl' --file-name "leADS_ey_symbionts" --model-name "leADS_ey" --rsfolder "leADS_ey_symbionts" --batch 50 --num-jobs 1
# python main.py --predict --binarize --pathway-report --no-parse --no-build-features --pred-labels --soft-voting --decision-threshold 0.5 --object-name "biocyc.pkl" --pathway2ec-idx-name "pathway2ec_idx.pkl" --pathway2ec-name 'pathway2ec.pkl' --hin-name 'hin_cmt.pkl' --features-name 'path2vec_cmt_tf_embeddings.npz' --X-name 'hots_4_Xe.pkl' --file-name "leADS_ey_hots" --model-name "leADS_ey" --rsfolder "leADS_ey_hots" --batch 50 --num-jobs 1
python main.py --predict --binarize --pred-labels --soft-voting --decision-threshold 0.5 --object-name "biocyc.pkl" --X-name 'biocyc21_tier3_9392_Xe.pkl' --file-name "biocyc21_tier3_9392_ey" --model-name "leADS_ey" --batch 50 --num-jobs 15

# python main.py --train --train-labels --binarize --use-external-features --calc-ads --ads-percent 0.7 --acquisition-type "mutual" --ssample-input-size 1.0 --ssample-label-size 2000 --calc-subsample-size 1000 --lambdas 0.01 0.01 0.01 0.01 0.01 10 --penalty "l21" --X-name "biocyc21_Xe.pkl" --y-name "biocyc21_y.pkl" --model-name "leADS_ml" --batch 50 --num-models 10 --num-epochs 10 --num-jobs 15 --display-interval 1
python main.py --evaluate --binarize --pred-labels --soft-voting --decision-threshold 0.5 --X-name "golden_Xe.pkl" --y-name "golden_y.pkl" --dsname "golden" --file-name "leADS_ml_1" --model-name "leADS_ml" --num-jobs 15
python main.py --evaluate --binarize --pred-labels --soft-voting --decision-threshold 0.5 --X-name "cami_Xe.pkl" --y-name "cami_y.pkl" --dsname "cami" --file-name "leADS_ml_2" --model-name "leADS_ml" --num-jobs 15
python main.py --evaluate --binarize --pred-labels --soft-voting --decision-threshold 0.5 --X-name "biocyc21_tier3_9392_Xe.pkl" --y-name "biocyc21_tier3_9392_y.pkl" --dsname "biocyc" --file-name "leADS_ml_3" --model-name "leADS_ml" --num-jobs 15
# python main.py --predict --binarize --pathway-report --no-parse --no-build-features --pred-labels --soft-voting --decision-threshold 0.5 --object-name "biocyc.pkl" --pathway2ec-idx-name "pathway2ec_idx.pkl" --pathway2ec-name 'pathway2ec.pkl' --hin-name 'hin_cmt.pkl' --features-name 'path2vec_cmt_tf_embeddings.npz' --X-name 'symbionts_Xe.pkl' --file-name "leADS_ml_symbionts" --model-name "leADS_ml" --rsfolder "leADS_ml_symbionts" --batch 50 --num-jobs 1
# python main.py --predict --binarize --pathway-report --no-parse --no-build-features --pred-labels --soft-voting --decision-threshold 0.5 --object-name "biocyc.pkl" --pathway2ec-idx-name "pathway2ec_idx.pkl" --pathway2ec-name 'pathway2ec.pkl' --hin-name 'hin_cmt.pkl' --features-name 'path2vec_cmt_tf_embeddings.npz' --X-name 'hots_4_Xe.pkl' --file-name "leADS_ml_hots" --model-name "leADS_ml" --rsfolder "leADS_ml_hots" --batch 50 --num-jobs 1
python main.py --predict --binarize --pred-labels --soft-voting --decision-threshold 0.5 --object-name "biocyc.pkl" --X-name 'biocyc21_tier3_9392_Xe.pkl' --file-name "biocyc21_tier3_9392_ml" --model-name "leADS_ml" --batch 50 --num-jobs 15

# python main.py --train --train-labels --binarize --use-external-features --calc-ads --ads-percent 0.7 --acquisition-type "variation" --top-k 50 --ssample-input-size 1.0 --ssample-label-size 2000 --calc-subsample-size 1000 --lambdas 0.01 0.01 0.01 0.01 0.01 10 --penalty "l21" --X-name "biocyc21_Xe.pkl" --y-name "biocyc21_y.pkl" --model-name "leADS_vr" --batch 50 --num-models 10 --num-epochs 10 --num-jobs 15 --display-interval 1
python main.py --evaluate --binarize --pred-labels --soft-voting --decision-threshold 0.5 --X-name "golden_Xe.pkl" --y-name "golden_y.pkl" --dsname "golden" --file-name "leADS_vr_1" --model-name "leADS_vr" --num-jobs 15
python main.py --evaluate --binarize --pred-labels --soft-voting --decision-threshold 0.5 --X-name "cami_Xe.pkl" --y-name "cami_y.pkl" --dsname "cami" --file-name "leADS_vr_2" --model-name "leADS_vr" --num-jobs 15
python main.py --evaluate --binarize --pred-labels --soft-voting --decision-threshold 0.5 --X-name "biocyc21_tier3_9392_Xe.pkl" --y-name "biocyc21_tier3_9392_y.pkl" --dsname "biocyc" --file-name "leADS_vr_3" --model-name "leADS_vr" --num-jobs 15
# python main.py --predict --binarize --pathway-report --no-parse --no-build-features --pred-labels --soft-voting --decision-threshold 0.5 --object-name "biocyc.pkl" --pathway2ec-idx-name "pathway2ec_idx.pkl" --pathway2ec-name 'pathway2ec.pkl' --hin-name 'hin_cmt.pkl' --features-name 'path2vec_cmt_tf_embeddings.npz' --X-name 'symbionts_Xe.pkl' --file-name "leADS_vr_symbionts" --model-name "leADS_vr" --rsfolder "leADS_vr_symbionts" --batch 50 --num-jobs 1
# python main.py --predict --binarize --pathway-report --no-parse --no-build-features --pred-labels --soft-voting --decision-threshold 0.5 --object-name "biocyc.pkl" --pathway2ec-idx-name "pathway2ec_idx.pkl" --pathway2ec-name 'pathway2ec.pkl' --hin-name 'hin_cmt.pkl' --features-name 'path2vec_cmt_tf_embeddings.npz' --X-name 'hots_4_Xe.pkl' --file-name "leADS_vr_hots" --model-name "leADS_vr" --rsfolder "leADS_vr_hots" --batch 50 --num-jobs 1
python main.py --predict --binarize --pred-labels --soft-voting --decision-threshold 0.5 --object-name "biocyc.pkl" --X-name 'biocyc21_tier3_9392_Xe.pkl' --file-name "biocyc21_tier3_9392_vr" --model-name "leADS_vr" --batch 50 --num-jobs 15

# python main.py --train --train-labels --binarize --use-external-features --calc-ads --ads-percent 0.7 --acquisition-type "psp" --top-k 50 --ssample-input-size 1.0 --ssample-label-size 2000 --calc-subsample-size 1000 --lambdas 0.01 0.01 0.01 0.01 0.01 10 --penalty "l21" --X-name "biocyc21_Xe.pkl" --y-name "biocyc21_y.pkl" --model-name "leADS_pp" --batch 50 --num-models 10 --num-epochs 10 --num-jobs 15 --display-interval 1
python main.py --evaluate --binarize --pred-labels --soft-voting --decision-threshold 0.5 --X-name "golden_Xe.pkl" --y-name "golden_y.pkl" --dsname "golden" --file-name "leADS_pp_1" --model-name "leADS_pp" --num-jobs 15
python main.py --evaluate --binarize --pred-labels --soft-voting --decision-threshold 0.5 --X-name "cami_Xe.pkl" --y-name "cami_y.pkl" --dsname "cami" --file-name "leADS_pp_2" --model-name "leADS_pp" --num-jobs 15
python main.py --evaluate --binarize --pred-labels --soft-voting --decision-threshold 0.5 --X-name "biocyc21_tier3_9392_Xe.pkl" --y-name "biocyc21_tier3_9392_y.pkl" --dsname "biocyc" --file-name "leADS_pp_3" --model-name "leADS_pp" --num-jobs 15
# python main.py --predict --binarize --pathway-report --no-parse --no-build-features --pred-labels --soft-voting --decision-threshold 0.5 --object-name "biocyc.pkl" --pathway2ec-idx-name "pathway2ec_idx.pkl" --pathway2ec-name 'pathway2ec.pkl' --hin-name 'hin_cmt.pkl' --features-name 'path2vec_cmt_tf_embeddings.npz' --X-name 'symbionts_Xe.pkl' --file-name "leADS_pp_symbionts" --model-name "leADS_pp" --rsfolder "leADS_pp_symbionts" --batch 50 --num-jobs 1
# python main.py --predict --binarize --pathway-report --no-parse --no-build-features --pred-labels --soft-voting --decision-threshold 0.5 --object-name "biocyc.pkl" --pathway2ec-idx-name "pathway2ec_idx.pkl" --pathway2ec-name 'pathway2ec.pkl' --hin-name 'hin_cmt.pkl' --features-name 'path2vec_cmt_tf_embeddings.npz' --X-name 'hots_4_Xe.pkl' --file-name "leADS_pp_hots" --model-name "leADS_pp" --rsfolder "leADS_pp_hots" --batch 50 --num-jobs 1
python main.py --predict --binarize --pred-labels --soft-voting --decision-threshold 0.5 --object-name "biocyc.pkl" --X-name 'biocyc21_tier3_9392_Xe.pkl' --file-name "biocyc21_tier3_9392_pp" --model-name "leADS_pp" --batch 50 --num-jobs 15

# train using random allocation
# python main.py --train --train-labels --binarize --use-external-features --ssample-input-size 0.7 --ssample-label-size 2000 --calc-subsample-size 1000 --lambdas 0.01 0.01 0.01 0.01 0.01 10 --penalty "l21" --X-name "biocyc21_Xe.pkl" --y-name "biocyc21_y.pkl" --model-name "leADS_ra" --batch 50 --num-models 10 --num-epochs 10 --num-jobs 15 --display-interval 1
python main.py --evaluate --binarize --pred-labels --soft-voting --decision-threshold 0.5 --X-name "golden_Xe.pkl" --y-name "golden_y.pkl" --dsname "golden" --file-name "leADS_ra_1" --model-name "leADS_ra" --num-jobs 15
python main.py --evaluate --binarize --pred-labels --soft-voting --decision-threshold 0.5 --X-name "cami_Xe.pkl" --y-name "cami_y.pkl" --dsname "cami" --file-name "leADS_ra_2" --model-name "leADS_ra" --num-jobs 15
python main.py --evaluate --binarize --pred-labels --soft-voting --decision-threshold 0.5 --X-name "biocyc21_tier3_9392_Xe.pkl" --y-name "biocyc21_tier3_9392_y.pkl" --dsname "biocyc" --file-name "leADS_ra_3" --model-name "leADS_ra" --num-jobs 15
# python main.py --predict --binarize --pathway-report --no-parse --no-build-features --pred-labels --soft-voting --decision-threshold 0.5 --object-name "biocyc.pkl" --pathway2ec-idx-name "pathway2ec_idx.pkl" --pathway2ec-name 'pathway2ec.pkl' --hin-name 'hin_cmt.pkl' --features-name 'path2vec_cmt_tf_embeddings.npz' --X-name 'symbionts_Xe.pkl' --file-name "leADS_ra_symbionts" --model-name "leADS_ra" --rsfolder "leADS_ra_symbionts" --batch 50 --num-jobs 1
# python main.py --predict --binarize --pathway-report --no-parse --no-build-features --pred-labels --soft-voting --decision-threshold 0.5 --object-name "biocyc.pkl" --pathway2ec-idx-name "pathway2ec_idx.pkl" --pathway2ec-name 'pathway2ec.pkl' --hin-name 'hin_cmt.pkl' --features-name 'path2vec_cmt_tf_embeddings.npz' --X-name 'hots_4_Xe.pkl' --file-name "leADS_ra_hots" --model-name "leADS_ra" --rsfolder "leADS_ra_hots" --batch 50 --num-jobs 1
python main.py --predict --binarize --pred-labels --soft-voting --decision-threshold 0.5 --object-name "biocyc.pkl" --X-name 'biocyc21_tier3_9392_Xe.pkl' --file-name "biocyc21_tier3_9392_ra" --model-name "leADS_ra" --batch 50 --num-jobs 15

# train without ads
# python main.py --train --train-labels --binarize --use-external-features --ssample-input-size 1.0 --ssample-label-size 2000 --calc-subsample-size 1000 --lambdas 0.01 0.01 0.01 0.01 0.01 10 --penalty "l21" --X-name "biocyc21_Xe.pkl" --y-name "biocyc21_y.pkl" --model-name "leADS" --batch 50 --num-models 10 --num-epochs 10 --num-jobs 15 --display-interval 1
python main.py --evaluate --binarize --pred-labels --soft-voting --decision-threshold 0.5 --X-name "golden_Xe.pkl" --y-name "golden_y.pkl" --dsname "golden" --file-name "leADS_fl_1" --model-name "leADS" --num-jobs 15
python main.py --evaluate --binarize --pred-labels --soft-voting --decision-threshold 0.5 --X-name "cami_Xe.pkl" --y-name "cami_y.pkl" --dsname "cami" --file-name "leADS_fl_2" --model-name "leADS" --num-jobs 15
python main.py --evaluate --binarize --pred-labels --soft-voting --decision-threshold 0.5 --X-name "biocyc21_tier3_9392_Xe.pkl" --y-name "biocyc21_tier3_9392_y.pkl" --dsname "biocyc" --file-name "leADS_fl_3" --model-name "leADS" --num-jobs 15
# python main.py --predict --binarize --pathway-report --no-parse --no-build-features --pred-labels --soft-voting --decision-threshold 0.5 --object-name "biocyc.pkl" --pathway2ec-idx-name "pathway2ec_idx.pkl" --pathway2ec-name 'pathway2ec.pkl' --hin-name 'hin_cmt.pkl' --features-name 'path2vec_cmt_tf_embeddings.npz' --X-name 'symbionts_Xe.pkl' --file-name "leADS_symbionts" --model-name "leADS" --rsfolder "leADS_symbionts" --batch 50 --num-jobs 1
# python main.py --predict --binarize --pathway-report --no-parse --no-build-features --pred-labels --soft-voting --decision-threshold 0.5 --object-name "biocyc.pkl" --pathway2ec-idx-name "pathway2ec_idx.pkl" --pathway2ec-name 'pathway2ec.pkl' --hin-name 'hin_cmt.pkl' --features-name 'path2vec_cmt_tf_embeddings.npz' --X-name 'hots_4_Xe.pkl' --file-name "leADS_hots" --model-name "leADS" --rsfolder "leADS_hots" --batch 50 --num-jobs 1
python main.py --predict --binarize --pred-labels --soft-voting --decision-threshold 0.5 --object-name "biocyc.pkl" --X-name 'biocyc21_tier3_9392_Xe.pkl' --file-name "biocyc21_tier3_9392_fl" --model-name "leADS" --batch 50 --num-jobs 15

#######################################################################################################
###################################           First test            ###################################
#######################################################################################################
### Train by varying models size in the ensemble
### Shamwow
### Fluctuate alot and we pick between 50 in [10, 50]
# EXPERIMENTS=(5 10 15 20 30 40 50 70 90 100)
# for p in ${!EXPERIMENTS[@]}; do
#     python main.py --train --train-labels --binarize --use-external-features --calc-ads --ads-percent 0.3 --acquisition-type "variation" --top-k ${EXPERIMENTS[p]} --ssample-input-size 1.0 --ssample-label-size 500 --calc-subsample-size 1000 --lambdas 0.01 0.01 0.01 0.01 0.01 10 --penalty "l21" --X-name "biocyc21_Xe.pkl" --y-name "biocyc21_y.pkl" --model-name "leADS_vm_vr_${p}" --batch 50 --num-models 3 --num-epochs 10 --num-jobs 12 --display-interval 1
#     python main.py --evaluate --pred-labels --soft-voting --decision-threshold 0.5 --X-name "golden_Xe.pkl" --y-name "golden_y.pkl" --dsname "golden" --file-name "leADS_vm_vr_${p}_1" --model-name "leADS_vm_vr_${p}" --num-jobs 1
#     python main.py --evaluate --pred-labels --soft-voting --decision-threshold 0.5 --X-name "cami_Xe.pkl" --y-name "cami_y.pkl" --dsname "cami" --file-name "leADS_vm_vr_${p}_2" --model-name "leADS_vm_vr_${p}" --num-jobs 1
# done
# 
# EXPERIMENTS=(5 10 15 20 30 40 50 70 90 100)
# for p in ${!EXPERIMENTS[@]}; do
#     python main.py --train --train-labels --binarize --use-external-features --calc-ads --ads-percent 0.3 --acquisition-type "psp" --top-k ${EXPERIMENTS[p]} --ssample-input-size 1.0 --ssample-label-size 500 --calc-subsample-size 1000 --lambdas 0.01 0.01 0.01 0.01 0.01 10 --penalty "l21" --X-name "biocyc21_Xe.pkl" --y-name "biocyc21_y.pkl" --model-name "leADS_vm_pp_${p}" --batch 50 --num-models 3 --num-epochs 10 --num-jobs 12 --display-interval 1
#     python main.py --evaluate --pred-labels --soft-voting --decision-threshold 0.5 --X-name "golden_Xe.pkl" --y-name "golden_y.pkl" --dsname "golden" --file-name "leADS_vm_pp_${p}_1" --model-name "leADS_vm_pp_${p}" --num-jobs 1
#     python main.py --evaluate --pred-labels --soft-voting --decision-threshold 0.5 --X-name "cami_Xe.pkl" --y-name "cami_y.pkl" --dsname "cami" --file-name "leADS_vm_pp_${p}_2" --model-name "leADS_vm_pp_${p}" --num-jobs 1
# done

#######################################################################################################
###################################           Second test           ###################################
#######################################################################################################
### Vary --ads-percent {0.3, 0.5, 0.7}
### Shamwow
# EXPERIMENTS=(0.3 0.5 0.7)
# for p in ${!EXPERIMENTS[@]}; do
#     python main.py --train --train-labels --binarize --use-external-features --calc-ads --ads-percent ${EXPERIMENTS[p]} --acquisition-type "entropy" --ssample-input-size 1.0 --ssample-label-size 500 --calc-subsample-size 1000 --lambdas 0.01 0.01 0.01 0.01 0.01 10 --penalty "l21" --X-name "biocyc21_Xe.pkl" --y-name "biocyc21_y.pkl" --model-name "leADS_ad_ey_${p}" --batch 50 --num-models 3 --num-epochs 5 --num-jobs 10 --display-interval 1
#     python main.py --evaluate --pred-labels --soft-voting --decision-threshold 0.5 --X-name "golden_Xe.pkl" --y-name "golden_y.pkl" --dsname "golden" --file-name "leADS_ad_ey_${p}_1" --model-name "leADS_ad_ey_${p}" --num-jobs 10
#     python main.py --evaluate --pred-labels --soft-voting --decision-threshold 0.5 --X-name "cami_Xe.pkl" --y-name "cami_y.pkl" --dsname "cami" --file-name "leADS_ad_ey_${p}_2" --model-name "leADS_ad_ey_${p}" --num-jobs 10
# 
#     python main.py --train --train-labels --binarize --use-external-features --calc-ads --ads-percent ${EXPERIMENTS[p]} --acquisition-type "mutual" --ssample-input-size 1.0 --ssample-label-size 500 --calc-subsample-size 1000 --lambdas 0.01 0.01 0.01 0.01 0.01 10 --penalty "l21" --X-name "biocyc21_Xe.pkl" --y-name "biocyc21_y.pkl" --model-name "leADS_ad_ml_${p}" --batch 50 --num-models 3 --num-epochs 5 --num-jobs 10 --display-interval 1
#     python main.py --evaluate --pred-labels --soft-voting --decision-threshold 0.5 --X-name "golden_Xe.pkl" --y-name "golden_y.pkl" --dsname "golden" --file-name "leADS_ad_ml_${p}_1" --model-name "leADS_ad_ml_${p}" --num-jobs 10
#     python main.py --evaluate --pred-labels --soft-voting --decision-threshold 0.5 --X-name "cami_Xe.pkl" --y-name "cami_y.pkl" --dsname "cami" --file-name "leADS_ad_ml_${p}_2" --model-name "leADS_ad_ml_${p}" --num-jobs 10
# 
#     python main.py --train --train-labels --binarize --use-external-features --calc-ads --ads-percent ${EXPERIMENTS[p]} --acquisition-type "variation" --top-k 50 --ssample-input-size 1.0 --ssample-label-size 500 --calc-subsample-size 1000 --lambdas 0.01 0.01 0.01 0.01 0.01 10 --penalty "l21" --X-name "biocyc21_Xe.pkl" --y-name "biocyc21_y.pkl" --model-name "leADS_ad_vr_${p}" --batch 50 --num-models 3 --num-epochs 5 --num-jobs 10 --display-interval 1
#     python main.py --evaluate --pred-labels --soft-voting --decision-threshold 0.5 --X-name "golden_Xe.pkl" --y-name "golden_y.pkl" --dsname "golden" --file-name "leADS_ad_vr_${p}_1" --model-name "leADS_ad_vr_${p}" --num-jobs 10
#     python main.py --evaluate --pred-labels --soft-voting --decision-threshold 0.5 --X-name "cami_Xe.pkl" --y-name "cami_y.pkl" --dsname "cami" --file-name "leADS_ad_vr_${p}_2" --model-name "leADS_ad_vr_${p}" --num-jobs 10
# 
#     python main.py --train --train-labels --binarize --use-external-features --calc-ads --ads-percent ${EXPERIMENTS[p]} --acquisition-type "psp" --top-k 50 --ssample-input-size 1.0 --ssample-label-size 500 --calc-subsample-size 1000 --lambdas 0.01 0.01 0.01 0.01 0.01 10 --penalty "l21" --X-name "biocyc21_Xe.pkl" --y-name "biocyc21_y.pkl" --model-name "leADS_ad_pp_${p}" --batch 50 --num-models 3 --num-epochs 5 --num-jobs 10 --display-interval 1
#     python main.py --evaluate --pred-labels --soft-voting --decision-threshold 0.5 --X-name "golden_Xe.pkl" --y-name "golden_y.pkl" --dsname "golden" --file-name "leADS_ad_pp_${p}_1" --model-name "leADS_ad_pp_${p}" --num-jobs 10
#     python main.py --evaluate --pred-labels --soft-voting --decision-threshold 0.5 --X-name "cami_Xe.pkl" --y-name "cami_y.pkl" --dsname "cami" --file-name "leADS_ad_pp_${p}_2" --model-name "leADS_ad_pp_${p}" --num-jobs 10

    # train using random allocation
#     python main.py --train --train-labels --binarize --use-external-features --ssample-input-size ${EXPERIMENTS[p]} --ssample-label-size 500 --calc-subsample-size 1000 --lambdas 0.01 0.01 0.01 0.01 0.01 10 --penalty "l21" --X-name "biocyc21_Xe.pkl" --y-name "biocyc21_y.pkl" --model-name "leADS_ra_${p}" --batch 50 --num-models 3 --num-epochs 5 --num-jobs 10 --display-interval 1
#     python main.py --evaluate --pred-labels --soft-voting --decision-threshold 0.5 --X-name "golden_Xe.pkl" --y-name "golden_y.pkl" --dsname "golden" --file-name "leADS_ra_${p}_1" --model-name "leADS_ra_${p}" --num-jobs 10
#     python main.py --evaluate --pred-labels --soft-voting --decision-threshold 0.5 --X-name "cami_Xe.pkl" --y-name "cami_y.pkl" --dsname "cami" --file-name "leADS_ra_${p}_2" --model-name "leADS_ra_${p}" --num-jobs 10
# done


#######################################################################################################
###################################            Third test           ###################################
#######################################################################################################
### Train by varying models size in the ensemble
### Shamwow
# EXPERIMENTS=(1 2 3 5 10 15 20 50)
# for p in ${!EXPERIMENTS[@]}; do
#   python main.py --train --train-labels --binarize --use-external-features --calc-ads --ads-percent 0.3 --acquisition-type "entropy" --ssample-input-size 1.0 --ssample-label-size 500 --calc-subsample-size 1000 --lambdas 0.01 0.01 0.01 0.01 0.01 10 --penalty "l21" --X-name "biocyc21_Xe.pkl" --y-name "biocyc21_y.pkl" --model-name "leADS_nm_ey_${p}" --batch 50 --num-models ${EXPERIMENTS[p]} --num-epochs 3 --num-jobs 12 --display-interval 1
#   python main.py --evaluate --pred-labels --soft-voting --decision-threshold 0.5 --X-name "golden_Xe.pkl" --y-name "golden_y.pkl" --dsname "golden" --file-name "leADS_nm_ey_${p}_1" --model-name "leADS_nm_ey_${p}" --num-jobs 1
#   python main.py --evaluate --pred-labels --soft-voting --decision-threshold 0.5 --X-name "cami_Xe.pkl" --y-name "cami_y.pkl" --dsname "cami" --file-name "leADS_nm_ey_${p}_2" --model-name "leADS_nm_ey_${p}" --num-jobs 1
# 
#   python main.py --train --train-labels --binarize --use-external-features --calc-ads --ads-percent 0.3 --acquisition-type "mutual" --ssample-input-size 1.0 --ssample-label-size 500 --calc-subsample-size 1000 --lambdas 0.01 0.01 0.01 0.01 0.01 10 --penalty "l21" --X-name "biocyc21_Xe.pkl" --y-name "biocyc21_y.pkl" --model-name "leADS_nm_ml_${p}" --batch 50 --num-models ${EXPERIMENTS[p]} --num-epochs 3 --num-jobs 12 --display-interval 1
#   python main.py --evaluate --pred-labels --soft-voting --decision-threshold 0.5 --X-name "golden_Xe.pkl" --y-name "golden_y.pkl" --dsname "golden" --file-name "leADS_nm_ml_${p}_1" --model-name "leADS_nm_ml_${p}" --num-jobs 1
#   python main.py --evaluate --pred-labels --soft-voting --decision-threshold 0.5 --X-name "cami_Xe.pkl" --y-name "cami_y.pkl" --dsname "cami" --file-name "leADS_nm_ml_${p}_2" --model-name "leADS_nm_ml_${p}" --num-jobs 1
# 
#   python main.py --train --train-labels --binarize --use-external-features --calc-ads --ads-percent 0.3 --acquisition-type "variation" --top-k 50 --ssample-input-size 1.0 --ssample-label-size 500 --calc-subsample-size 1000 --lambdas 0.01 0.01 0.01 0.01 0.01 10 --penalty "l21" --X-name "biocyc21_Xe.pkl" --y-name "biocyc21_y.pkl" --model-name "leADS_nm_vr_${p}" --batch 50 --num-models ${EXPERIMENTS[p]} --num-epochs 3 --num-jobs 12 --display-interval 1
#   python main.py --evaluate --pred-labels --soft-voting --decision-threshold 0.5 --X-name "golden_Xe.pkl" --y-name "golden_y.pkl" --dsname "golden" --file-name "leADS_nm_vr_${p}_1" --model-name "leADS_nm_vr_${p}" --num-jobs 1
#   python main.py --evaluate --pred-labels --soft-voting --decision-threshold 0.5 --X-name "cami_Xe.pkl" --y-name "cami_y.pkl" --dsname "cami" --file-name "leADS_nm_vr_${p}_2" --model-name "leADS_nm_vr_${p}" --num-jobs 1
# 
#   python main.py --train --train-labels --binarize --use-external-features --calc-ads --ads-percent 0.3 --acquisition-type "psp" --top-k 50 --ssample-input-size 1.0 --ssample-label-size 500 --calc-subsample-size 1000 --lambdas 0.01 0.01 0.01 0.01 0.01 10 --penalty "l21" --X-name "biocyc21_Xe.pkl" --y-name "biocyc21_y.pkl" --model-name "leADS_nm_pp_${p}" --batch 50 --num-models ${EXPERIMENTS[p]} --num-epochs 3 --num-jobs 12 --display-interval 1
#   python main.py --evaluate --pred-labels --soft-voting --decision-threshold 0.5 --X-name "golden_Xe.pkl" --y-name "golden_y.pkl" --dsname "golden" --file-name "leADS_nm_pp_${p}_1" --model-name "leADS_nm_pp_${p}" --num-jobs 1
#   python main.py --evaluate --pred-labels --soft-voting --decision-threshold 0.5 --X-name "cami_Xe.pkl" --y-name "cami_y.pkl" --dsname "cami" --file-name "leADS_nm_pp_${p}_2" --model-name "leADS_nm_pp_${p}" --num-jobs 1

#     python main.py --train --train-labels --binarize --use-external-features --ssample-input-size 0.3 --ssample-label-size 500 --calc-subsample-size 1000 --lambdas 0.01 0.01 0.01 0.01 0.01 10 --penalty "l21" --X-name "biocyc21_Xe.pkl" --y-name "biocyc21_y.pkl" --model-name "leADS_nm_ra_${p}" --batch 50 --num-models ${EXPERIMENTS[p]} --num-epochs 3 --num-jobs 12 --display-interval 1
#     python main.py --evaluate --pred-labels --soft-voting --decision-threshold 0.5 --X-name "golden_Xe.pkl" --y-name "golden_y.pkl" --dsname "golden" --file-name "leADS_nm_ra_${p}_1" --model-name "leADS_nm_ra_${p}" --num-jobs 1
#     python main.py --evaluate --pred-labels --soft-voting --decision-threshold 0.5 --X-name "cami_Xe.pkl" --y-name "cami_y.pkl" --dsname "cami" --file-name "leADS_nm_ra_${p}_2" --model-name "leADS_nm_ra_${p}" --num-jobs 1
# done
