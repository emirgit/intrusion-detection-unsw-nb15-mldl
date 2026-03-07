import os
from kaggle.api.kaggle_api_extended import KaggleApi

"""
Note: Before running the script,
you will need to ensure your kaggle.json API token is placed
in ~/.kaggle/ as per the standard Kaggle API configuration.

You can obtain it from kaggle.com/settings
For more information, kaggle.com/docs/api or github.com/Kaggle/kaggle-cli
"""




# --- CONFIGURATION ---
# The Kaggle dataset slug (owner/dataset-name)
DATASET_SLUG = "dhoogla/unswnb15"

# Directory where the dataset will be downloaded and extracted
# Relative path to the project root
BASE_DOWNLOAD_DIR = "dataset/unsw-nb15-dataset/training-and-testing-parquet"


def setup_kaggle_api():
    """
    Authenticates with the Kaggle API.
    """
    try:
        api = KaggleApi()
        api.authenticate()
        return api
    except Exception as e:
        print(f"Error : Authentication failed. Reason: {e}")
        print("Hint  : Ensure 'kaggle.json' is located in your ~/.kaggle/ directory.")
        exit(1)


def main():
    """
    Downloads and extracts the UNSW-NB15 dataset.
    """
    # 1. Setup: Authenticate API
    api = setup_kaggle_api()

    # 2. Processing: Download and Extract
    target_dir = BASE_DOWNLOAD_DIR
    
    print(f"Info   : Target dataset: {DATASET_SLUG}")
    print(f"Status : Downloading and extracting to {target_dir}...")

    # Create the directory structure if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    try:
        # Download and unzip the files
        api.dataset_download_files(DATASET_SLUG, path=target_dir, unzip=True)
        print("Status : Download and extraction successful.")
    except Exception as e:
        print(f"Error  : Failed to download {DATASET_SLUG}. Reason: {e}")

    print("\nInfo   : Operation completed.")


if __name__ == "__main__":
    main()
