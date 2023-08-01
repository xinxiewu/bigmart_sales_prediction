"""
__maim__.py contains the workflow to run all sub-programs
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
from util import *

sns.set()

def main(fileurl=None, fileurl_unseen=None, output=None, target=None, discrete_x=None, continuos_x=None):
    """
    Step 1: Data Preparation & EDA
    """
    # 1.1 Download dataset from Github & read as DataFrame
    df = download_file(fileurl)
    df_unseen = download_file(fileurl_unseen)
    # 1.2 EDA
    print(f"Loaded modeling data {df.shape} and unseen data {df_unseen.shape}. \n\nEDA Starts")
    ## (a) Feature Distribution (Extreme Value) & Missing Value Detection

    
    



    return


if __name__ == '__main__':
    """
    Step 1: Clean output folder
    """
    delete_files()
    """
    Step 2: Call the main program
    """
    # main(fileurl = 'https://raw.githubusercontent.com/xinxiewu/datasets/main/bigmart_sales/Train.csv',
    #      fileurl_unseen = 'https://raw.githubusercontent.com/xinxiewu/datasets/main/bigmart_sales/Test.csv',
    #      output = r'../public/output',
    #      target = 'Item_Outlet_Sales',
    #      discrete_x = ('Item_Identifier', 'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 
    #                       'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'),
    #      continuos_x = ('Item_Weight', 'Item_Visibility', 'Item_MRP')
    #      )