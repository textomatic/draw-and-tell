import os
import subprocess


# Define constants
_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
_DATA_DIR = '../data'
_DATASET = 'train_simplified.zip' # 'train_raw.zip'


def main():
    """
    Downloads the Quick, Draw! Doodle Recognition Challenge dataset from Kaggle and decompresses it.
    Note: 
        - Only the simplified training data is downloaded, not the raw training data
        - Update the value of `_DATASET` if the raw data is preferred
        - Make sure your Kaggle API credentials are present in `~/.kaggle/kaggle.json`. If not, create an API token at `https://www.kaggle.com/<username>/account` and download the JSON file

    Args:
        None

    Returns:
        None
    """
    # Define download destination
    download_path = os.path.join(_CURRENT_DIR, _DATA_DIR)
    download_zip = os.path.join(download_path, _DATASET)

    # Execute command in shell to download dataset
    subprocess.call(f'kaggle competitions download -c quickdraw-doodle-recognition -f {_DATASET} -p {download_path}', shell=True)

    # Execute command in shell to decompress dataset
    subprocess.call(f'unzip -d {download_path} {download_zip}', shell=True)


if __name__ == '__main__':
    main()