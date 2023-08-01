"""
util.py contains custom functions:
    1. download_file: Download the .csv file from the given link and read as dataframe
    2. delete_files: Delete files in the given folder, except README.md
    3. feature_distribution: Generate plots of features distribution and save
"""
import requests
import pandas as pd
import os

# download_file(url, output)
def download_file(url=None, output=r'../public/output'):
    """ Download the .csv file from the given link and read as dataframe

    Args: 
        url: str
        output: path to store downloaded files
    
    Returns:
        DataFrame
    """
    local_filename = os.path.join(output, url.split('/')[-1])
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)
    return pd.read_csv(local_filename)

# delete_files(path, keep)
def delete_files(path=r'../public/output', keep=['README.md']):
    """ Delete files in the given folder path, except README.md

    Args:
        path: path, starting with r''
        keep: files to keep, default value as README.md

    Returns:
        nothing to return
    """
    for fname in os.listdir(path):
        if fname not in (keep):
            os.remove(os.path.join(path, fname))
    return

# feature_distribution(discrete_x, continuos_x, output)
def feature_distribution(df=None, discrete_x=None, continuos_x=None, target=None, 
                         output=r'../public/output', subplots=[4,3], figsize=(30,30)):
    """ Generate plots of features distribution and save

    Args:
        df: DataFrame
        discrete_x: set, discrete features
        continuous_x: set, continuous features
        target: str, dependent variables
        output: path to save outputs
        subplots: subplots
        figsize: figure size

    Returns:
        nothing to return
    """
    pass