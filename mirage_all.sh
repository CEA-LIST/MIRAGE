#!/bin/bash

./mirage_gnn_features_ablation.sh
./mirage_models_features_ablation.sh
./mirage_clean_training.sh
./mirage_clean_noisy_validation.sh
./mirage_gnn_tscale_ablation.sh