import visualization_utils
import argparse
import ast

def process_and_save_logs(folder_paths, csv_filename, test_folder_name="test"):
    """
    Process tensorboard logs from given folders and save the results to a csv file.

    Args:
        folder_paths (list): List of folder paths containing tensorboard logs.
        csv_filename (str): Path to the output csv file.
        test_folder_name (str, optional): Name of the test folder. Defaults to "test".
    """
    if "tscale" in csv_filename:
        group_keys = ["dataset_input_features", "dataset_posconv_features", "dataset_posgraph_features", "model_t_scale"]
    else:
        group_keys = ["dataset_input_features", "dataset_posconv_features", "dataset_posgraph_features"]

    # Output
    if test_folder_name == "val_noisy":
        summary_df = visualization_utils.process_tensorboard_logs(folder_paths, group_keys, test_folder_name
                ).rename(columns={"test_accuracy_mean_std": "Val_noisy_mean_std"}
                ).drop(columns=["max_accuracy_mean_std", "loss_mean_std", "val_loss_mean_std", "test_loss_mean_std", "test_top3_accuracy_mean_std"])
        
        summary_df.to_csv(csv_filename + "/summary_results_valnoisy.csv")

    elif test_folder_name == "val_clean":
        summary_df = visualization_utils.process_tensorboard_logs(folder_paths, group_keys, test_folder_name
                ).rename(columns={"test_accuracy_mean_std": "Val_clean_mean_std"}
                ).drop(columns=["max_accuracy_mean_std", "loss_mean_std", "val_loss_mean_std", "test_loss_mean_std", "test_top3_accuracy_mean_std"])
        
        summary_df.to_csv(csv_filename + "/summary_results_valclean.csv")

    else:
        summary_df = visualization_utils.process_tensorboard_logs(folder_paths, group_keys, test_folder_name).sort_values(
            by='max_val_accuracy_mean_std', ascending=True
        )
        summary_df.to_csv(csv_filename + "/summary_results.csv")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process tensorboard logs and save results to a csv file.')

    # Add arguments
    parser.add_argument('--csv_filename', required=True, help='Path to the output csv file')
    parser.add_argument('--test_folder_name', default='test', help='Name of the test folder (default: test. indicate "val_noisy" or "val_clean" for validation on clean-only or noise-only samples)')
    parser.add_argument('folder_paths', nargs='*', help='Paths to folders containing tensorboard logs')

    # Parse arguments
    args = parser.parse_args()

    # Call the processing function with the provided arguments
    process_and_save_logs(args.folder_paths, args.csv_filename, test_folder_name=args.test_folder_name)

if __name__ == "__main__":
    main()