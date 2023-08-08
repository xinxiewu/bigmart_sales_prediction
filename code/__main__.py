"""
__maim__.py contains the workflow to run all sub-programs
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
from util import *

def main(fileurl=None, fileurl_unseen=None, output=None, target=None, discrete_x=None, continuos_x=None):
    """
    Step 1: Data Preparation & EDA
    """
    # 1.1 Download dataset from Github & read as DataFrame
    df = download_file(fileurl)
    df_unseen = download_file(fileurl_unseen)
    # 1.2 EDA
    print(f"Loaded modeling data {df.shape} and unseen data {df_unseen.shape}. \n\nEDA Starts")
    ## (a) Feature Distribution, Extreme & Missing Value Detection
    missing, categorical = feature_distribution(df=df, discrete_x=discrete_x, continuos_x=continuos_x, target=target)
    categorical.add('Outlet_Establishment_Year')
    print(f"Features with missing values: {missing}\nCategorical features: {categorical}")
    ## (b) Feature Engineering 1: Missing Value & Categorical -> Numeric
    """
    Missing 1: Item_Weight - Same Item_Identifier has the same weight; Otherwise, avg of Item_Fat_Content and Item_Type
    Missing 2: Outlet_Size - Grocery Store is Small; Outlet017 is High; Outlet045 is Medium 
    Missing 3: Item_Visibility - Avg of Item_Identifier; Outlet_Identifier -> Item_Type -> Item_Fat_Content -> Avg
    """
    df = missing_handler(df=df, missing=missing, output=output)
    df = df.replace({'Item_Fat_Content':{'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'}})
    if df[getattr(df, 'Item_Weight').isna()==True]['Item_Identifier'].count() == 0 and df[getattr(df, 'Item_Visibility') == 0]['Item_Identifier'].count() == 0 and df[getattr(df, 'Outlet_Size').isna()==True]['Item_Identifier'].count() == 0:
        print("Fixed Missing Values.")
    else:
        print("Missing Values Exist.")
    """ Categorical -> Numeric
    Item_Identifier (1,559): Frequency -> Violinplot -> One-hot (dummy, drop first or last columns)
    Item_Fat_Content (4->2): Violinplot -> One-hot (dummy, drop first or last columns)
    Item_Type (16): Violinplot -> One-hot (dummy, drop first or last columns)
    Outlet_Identifier (10): Violinplot -> Frequency
    Outlet_Establishment_Year (9): Violinplot -> Frequency
    Outlet_Size (3): Violinplot -> Ordinal
    Outlet_Location_Type (3): Violinplot -> Ordinal
    Outlet_Type (4): Violinplot -> Ordinal
    """
    df = categorical_conversion(df=df, categorical_feature=categorical)

    ## (c) 


    return


if __name__ == '__main__':
    """
    Step 1: Clean output folder
    """
    delete_files()
    """
    Step 2: Call the main program
    """
    main(fileurl = 'https://raw.githubusercontent.com/xinxiewu/datasets/main/bigmart_sales/Train.csv',
         fileurl_unseen = 'https://raw.githubusercontent.com/xinxiewu/datasets/main/bigmart_sales/Test.csv',
         output = r'../public/output',
         target = 'Item_Outlet_Sales',
         discrete_x = ('Item_Identifier', 'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 
                          'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'),
         continuos_x = ('Item_Weight', 'Item_Visibility', 'Item_MRP')
         )