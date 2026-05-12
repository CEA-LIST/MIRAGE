#!/bin/bash

CONFIG=configs/mirage.toml
model_list=("pointnet_tiny" "transformer_tiny" "cnn_tiny")

## train (and eval) only on clean samples

for model in "${model_list[@]}"; do
    save_name=lightning_logs/clean_training_${model}
    for ((kfold_id=0; kfold_id<5; kfold_id++)); do
        PROCESSED_DATA="data/MIRAGE/processed/kfold${kfold_id}_evalclean.pkl"
        ## model training
        python mm models_clean_training mirage $model -config ${CONFIG} -d_kfold $kfold_id -seed $kfold_id -d_processed_data $PROCESSED_DATA -save $save_name
    done
    ## create csv file with results (NOTE: train/val/test accuracy is on clean only samples)
    python mmrnet/utils/process_logs.py --csv_filename $save_name --test_folder_name test ${save_name}/tuning_*
done