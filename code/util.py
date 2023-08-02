"""
util.py contains custom functions:
    1. download_file: Download the .csv file from the given link and read as dataframe
    2. delete_files: Delete files in the given folder, except README.md
    3. feature_distribution: Generate plots of features distribution and save
    4. missing_handler: Handle missing values
    5. categorical_conversion: Convert categorical to numeric
"""
import requests
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

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
def feature_distribution(df=None, discrete_x=None, continuos_x=None, target=None, fname='variables_dist',
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
        missing: set, variables with missing values
        categorical: set, variables with categorical
    """
    missing, categorical = set(), set()
    sns.set()
    fig, axes = plt.subplots(subplots[0], subplots[1], figsize=figsize)
    ax = axes.flatten()
    for i, col in enumerate(df.columns):
        if col in discrete_x:
            data_type, data_cat = 'Discrete', 'Numeric' if col == 'Outlet_Establishment_Year' else 'Categorical'
            if data_cat == 'Categorical':
                categorical.add(col)
            miss = df[col].isna().value_counts()[True] if True in df[col].isna().value_counts().index else 0
            if miss > 0:
                missing.add(col)
            if col == 'Item_Identifier':
                disct = sns.countplot(data=pd.DataFrame(df[col].value_counts()), x=col,ax=ax[i])
                disct.set(xlabel=f"Appearance of {col}: {data_type}+{data_cat}", ylabel='Count of Items', title=f"{col} - {miss} Missing")
            else:
                disct = sns.countplot(data=df, x=col,ax=ax[i])
                disct.set(xlabel=f"{col}: {data_type}+{data_cat}", ylabel='Count', title=f"{col} - {miss} Missing")
                if col in ('Item_Type', 'Outlet_Identifier', 'Outlet_Establishment_Year'):
                    disct.set_xticklabels(disct.get_xticklabels(), rotation=30, fontsize=10)
        elif col in continuos_x or col == target:
            data_type, data_cat = 'Continuous', 'Numeric'
            if col == 'Item_Visibility':
                miss = (getattr(df,col)==0).value_counts()[True] if True in (getattr(df,col)==0).value_counts().index else 0
                cont = sns.histplot(df[df[col]!=0][col],kde=True,color='purple',ax=ax[i])
            else:
                miss = df[col].isna().value_counts()[True] if True in df[col].isna().value_counts().index else 0
                cont = sns.histplot(getattr(df,col),kde=True,color='purple',ax=ax[i])
            cont.set(xlabel=f"{col}: {data_type}+{data_cat}", ylabel='Density', title=f"{col} - {miss} Missing")
            if miss > 0:
                missing.add(col)
    plt.tight_layout()
    plt.savefig(os.path.join(output, fname))
    return missing, categorical

# missing_handler()
def missing_handler(df=None, missing=None, output=r'../public/output', fname='missing_variables.png'):
    """ Handle features with missing values

    Args:
        df: DataFrame
        missing: set, feature names with missing values
        output: path 
        fname: output file name

    Returns:
        
    """
    return df

# categorical_conversion()
def categorical_conversion(df=None, categorical_feature=None):
    return df