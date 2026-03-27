import os

def rename_folders_recursively(root_dir):
    # Define the mapping of old folder names to new folder names
    folder_mapping = {
        "auf": "gesture0",
        "ci": "gesture1",
        "co": "gesture2",
        "ct": "gesture3",
        "dci": "gesture4",
        "dco": "gesture5",
        "pci": "gesture6",
        "pco": "gesture7",
        "pp": "gesture8",
        "rf": "gesture9",
        "rs": "gesture10",
        "wr": "gesture11"
    }

    # Walk through the directory tree
    for root, dirs, files in os.walk(root_dir):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if dir_name in folder_mapping:
                # Construct the new folder name
                new_folder_name = folder_mapping[dir_name]
                new_folder_path = os.path.join(root, new_folder_name)

                # Rename the folder
                os.rename(dir_path, new_folder_path)
                print(f"Renamed: {dir_path} -> {new_folder_path}")

if __name__ == "__main__":
    # Specify the path to the directory containing the subfolders
    root_directory = input("Enter the path to the directory containing the subfolders: ")
    rename_folders_recursively(root_directory)