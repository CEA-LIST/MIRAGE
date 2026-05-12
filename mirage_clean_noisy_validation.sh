#!/bin/bash

CONFIG=configs/mirage.toml
model_list=("pointnet_tiny" "transformer_tiny" "cnn_tiny")


#### for models trained on clean only samples #######
## validation on noisy samples (validation on clean samples already done in the clean training)
## NOTE: models must have been trained before using ./mirage_clean_training.sh

for model in "${model_list[@]}"; do
    save_name=lightning_logs/clean_training_${model}
    
    for ((kfold_id=0; kfold_id<5; kfold_id++)); do

        for VERSION_DIR in "${save_name}/tuning_${kfold_id}"/*/; do

            ## recover the hyperparameters (posconv_features and input_features) used in this version (stored in hparams.yaml)
            INPUT=$(grep 'dataset_input_features:' "${VERSION_DIR}/hparams.yaml" | awk -F': ' '{print $2}' | tr -d '\n' | sed "s/'//g")
            POSCONV=$(grep 'dataset_posconv_features:' "${VERSION_DIR}/hparams.yaml" | awk -F': ' '{print $2}' | tr -d '\n' | sed "s/'//g")

            ## evaluate on noise samples of the validation set
            PROCESSED_DATA="data/MIRAGE/processed/kfold${kfold_id}_evalnoisy.pkl"
            python mm evalnoisy mirage $model -load "$VERSION_DIR" -config ${CONFIG} -d_kfold $kfold_id -seed $kfold_id -d_processed_data $PROCESSED_DATA -d_input_features "$INPUT" -d_posconv_features "$POSCONV"
        done

    done

    ## create csv file with results "summary_results.csv"
    python mmrnet/utils/process_logs.py --csv_filename $save_name --test_folder_name val_noisy ${save_name}/tuning_*

done



#### for models trained on clean+noisy samples #########
## validation on clean and then eval on noisy samples
## NOTE: models must have been trained before using ./mirage_models_features_ablation.sh

for model in "${model_list[@]}"; do
    save_name=lightning_logs/models_features_${model}
    
    for ((kfold_id=0; kfold_id<5; kfold_id++)); do

        for VERSION_DIR in "${save_name}/tuning_${kfold_id}"/*/; do

            ## recover the hyperparameters (posconv_features and input_features) used in this version (stored in hparams.yaml)
            INPUT=$(grep 'dataset_input_features:' "${VERSION_DIR}/hparams.yaml" | awk -F': ' '{print $2}' | tr -d '\n' | sed "s/'//g")
            POSCONV=$(grep 'dataset_posconv_features:' "${VERSION_DIR}/hparams.yaml" | awk -F': ' '{print $2}' | tr -d '\n' | sed "s/'//g")

            ## evaluate on noise samples of the validation set
            PROCESSED_DATA="./data/MIRAGE/processed/kfold${kfold_id}_evalnoisy.pkl"
            python mm evalnoisy mirage $model -load "$VERSION_DIR" -config ${CONFIG} -d_kfold $kfold_id -seed $kfold_id -d_processed_data $PROCESSED_DATA  -d_posconv_features "$POSCONV" -d_input_features "$INPUT"

            ## evaluate on clean samples of the validation set
            PROCESSED_DATA="./data/MIRAGE/processed/kfold${kfold_id}_evalclean.pkl"
            python mm evalnoisy mirage $model -load "$VERSION_DIR" -config ${CONFIG} -d_kfold $kfold_id -seed $kfold_id -d_processed_data $PROCESSED_DATA  -d_posconv_features "$POSCONV" -d_input_features "$INPUT"
        done

    done
    
    ## create csv file with results  "summary_results_valnoisy.csv"
    python mmrnet/utils/process_logs.py --csv_filename $save_name --test_folder_name val_noisy ${save_name}/tuning_*
    ## create csv file with results  "summary_results_valclean.csv"
    python mmrnet/utils/process_logs.py --csv_filename $save_name --test_folder_name val_clean ${save_name}/tuning_*
done