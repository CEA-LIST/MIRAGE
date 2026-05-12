#!/bin/bash

CONFIG=configs/mirage.toml
model_list=("pointnet_tiny")

for model in "${model_list[@]}"; do
    save_name=lightning_logs/gnn_tscale
    for ((kfold_id=0; kfold_id<5; kfold_id++)); do
        PROCESSED_DATA="data/MIRAGE/processed/kfold${kfold_id}.pkl"
        ## model training
        python mm gnn_tscale_ablation mirage $model -config ${CONFIG} -d_kfold $kfold_id -seed $kfold_id -d_processed_data $PROCESSED_DATA -save $save_name
    done
    ## create csv file with results
    python mmrnet/utils/process_logs.py --csv_filename $save_name --test_folder_name test ${save_name}/tuning_*
done