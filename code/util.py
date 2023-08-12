"""
util.py contains custom functions:
    1. download_file: Download the .csv file from the given link and read as dataframe
    2. delete_files: Delete files in the given folder, except README.md
    3. feature_distribution: Generate plots of features distribution and save
    4. missing_handler: Handle missing values
    5. categorical_conversion: Convert categorical to numeric
    6. numeric_conversion: Scaling
    7. data_split: Split dataset into training, validation and testing
    8. point_eval_metric: Given pred and actual y, generate evaluation metrics: R^2, EVS, MSE, RMSE, MAE
"""
import requests
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

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
def missing_handler(df=None, missing=None, output=r'../public/output', fname='missing_variables.png',
                    subplots=[3,5], figsize=(50,30)):
    """ Handle features with missing values

    Args:
        df: DataFrame
        missing: set, feature names with missing values
        output: path 
        fname: output file name

    Returns:
        
    """
    sns.set()
    fig, axes = plt.subplots(subplots[0], subplots[1], figsize=figsize)
    ax, i = axes.flatten(), 0
    for miss in missing:
        df_0 = df.copy()
        # Item_Weight -> Item_Identifier: Same value if have, otherwise industry mean (Fat & Type)
        if miss == 'Item_Weight':
            key = 'Item_Identifier'
            identifier_cnt_0, identifier_cnt_1, identifier_unique = getattr(df, key).unique(), getattr(df[getattr(df, miss).isna()==False], key).unique(), []
            for id in identifier_cnt_0:
                if id not in identifier_cnt_1:
                    identifier_unique.append(id)
            df_0[key] = pd.factorize(getattr(df,key))[0]
            grouped = pd.DataFrame(df_0.groupby(key)[miss].nunique().reset_index())
            overall = sns.lineplot(data=grouped, x=key, y=miss, ax=ax[i])
            overall.set(xlabel=f"{key}", ylabel=f"Distinct Values of {miss}", title=f"Line Chart of {miss} vs {key}")
            i += 1
            for item in identifier_unique:
                fat_cont = ['Regular','reg'] if item != 'FDK57' else ['LF', 'Low Fat']
                if item == 'FDN52':
                    item_type = 'Frozen Foods'
                elif item == 'FDK57':
                    item_type = 'Snack Foods'
                elif item == 'FDE52':
                    item_type = 'Dairy'
                else:
                    item_type = 'Baking Goods'
                temp = df.query(f"Item_Fat_Content in {fat_cont} & Item_Type == '{item_type}'")
                temp_0 = sns.histplot(getattr(temp, miss),kde=True,color='purple',ax=ax[i])
                temp_0.set(xlabel=f"{key} of {fat_cont} & {item_type}", ylabel=f"Weight Distribution", 
                           title=f"{miss} - Mean: {format(getattr(temp,miss).mean(),'.2f')}, Median: {format(getattr(temp,miss).median(),'.2f')}, STD: {format(getattr(temp,miss).std(),'.2f')}")
                i += 1
            df_t = df[df['Item_Weight'].isna()==False][['Item_Identifier', 'Item_Weight']].groupby(by=['Item_Identifier']).mean().reset_index()
            df_t = pd.concat([df_t, pd.DataFrame.from_dict({'Item_Identifier':['FDN52'], 'Item_Weight':[13.181]})], ignore_index=True)
            df_t = pd.concat([df_t, pd.DataFrame.from_dict({'Item_Identifier':['FDK57'], 'Item_Weight':[13.707]})], ignore_index=True)
            df_t = pd.concat([df_t, pd.DataFrame.from_dict({'Item_Identifier':['FDE52'], 'Item_Weight':[13.484]})], ignore_index=True)
            df_t = pd.concat([df_t, pd.DataFrame.from_dict({'Item_Identifier':['FDQ60'], 'Item_Weight':[12.013]})], ignore_index=True)
            df = df.merge(df_t, how='left', on='Item_Identifier').drop(columns=['Item_Weight_x']).rename(columns={'Item_Weight_y': 'Item_Weight'})
        # Item_Visibility -> Item_Identifier: Median
        elif miss == 'Item_Visibility':
            key, key_new = 'Item_Identifier', 'Item_No'
            df_0 = df_0[[miss, key]]
            df_0 = df_0[getattr(df, miss)!=0].sort_values(by=[miss,key])
            df_0[key_new] = pd.factorize(getattr(df_0,key))[0]
            avg = df_0.groupby(by=[key, key_new]).mean().reset_index().rename(columns={miss:'mean'})
            med = df_0.groupby(by=[key, key_new]).median().reset_index().rename(columns={miss:'median'})
            maxi = df_0.groupby(by=[key, key_new]).max().reset_index().rename(columns={miss:'max'})
            mini = df_0.groupby(by=[key, key_new]).min().reset_index().rename(columns={miss:'min'})
            final = avg.merge(med,how='left',on=[key, key_new]).merge(maxi,how='left',on=[key, key_new]).merge(mini,how='left',on=[key, key_new])
            final = final.sort_values(by=key_new).reset_index()
            temp = sns.lineplot(data=final[['mean','median','max','min']], ax=ax[i])
            temp.set(xlabel=f"{key} - factorized", ylabel=f"Trend", title=f"Line Chart of Overall Trend")
            i += 1
            temp = sns.lineplot(data=final,x=key_new,y='mean', ax=ax[i])
            temp.set(xlabel=f"{key} - factorized", ylabel=f"Trend", title=f"Line Chart of Mean Trend")
            i += 1
            temp = sns.lineplot(data=final,x=key_new,y='median', ax=ax[i])
            temp.set(xlabel=f"{key} - factorized", ylabel=f"Trend", title=f"Line Chart of Median Trend")
            i += 1
            temp = sns.lineplot(data=final,x=key_new,y='min', ax=ax[i])
            temp.set(xlabel=f"{key} - factorized", ylabel=f"Trend", title=f"Line Chart of Min Trend")
            i += 1
            temp = sns.lineplot(data=final,x=key_new,y='max', ax=ax[i])
            temp.set(xlabel=f"{key} - factorized", ylabel=f"Trend", title=f"Line Chart of Max Trend")
            i += 1
            df_t = df[df['Item_Visibility'] != 0][['Item_Identifier', 'Item_Visibility']].groupby(by=['Item_Identifier']).mean().reset_index()
            df = df.merge(df_t, how='left', on='Item_Identifier').drop(columns=['Item_Visibility_x']).rename(columns={'Item_Visibility_y': 'Item_Visibility'})
        # Outlet_Size
        elif miss == 'Outlet_Size':
            miss = 'Outlet_Size'
            df_0 = df.copy()[['Outlet_Identifier', 'Outlet_Establishment_Year', miss, 'Outlet_Location_Type', 'Outlet_Type']]\
                .drop_duplicates().sort_values(by=['Outlet_Identifier']).reset_index().drop(columns=['index'])
            df_1 = df[['Outlet_Identifier', 'Item_Outlet_Sales']].groupby(by=['Outlet_Identifier']).sum(numeric_only=True).reset_index()
            df_1['Item_Outlet_Sales'] = df_1['Item_Outlet_Sales']//1000
            df_2 = df[['Outlet_Identifier', 'Item_Identifier']].groupby(by=['Outlet_Identifier']).count().reset_index()
            df_final = df_0.merge(df_1,how='left',on='Outlet_Identifier').merge(df_2,how='left',on='Outlet_Identifier')
            temp = sns.countplot(data=df_final,x='Outlet_Establishment_Year',hue='Outlet_Location_Type',ax=ax[i])
            temp.set(xlabel=f"Outlet_Establishment_Year", ylabel='Outlet_Location_Type', title=f"Outlet Establishment by Year & Location")
            temp.legend(loc='lower center')
            i += 1
            temp = sns.countplot(data=df_final,x='Outlet_Location_Type',hue='Outlet_Size',ax=ax[i])
            temp.set(xlabel=f"Outlet_Location_Type", ylabel='Outlet_Size', title=f"Outlet Distribution by Location & Size")
            temp.legend(loc='lower center')
            i += 1
            temp = sns.countplot(data=df_final,x='Outlet_Location_Type',hue='Outlet_Type',ax=ax[i])
            temp.set(xlabel=f"Outlet_Location_Type", ylabel='Outlet_Type', title=f"Outlet Distribution by Type & Location")
            temp.legend(loc='lower center')
            i += 1
            temp = sns.barplot(data=df_final,x='Outlet_Identifier',y='Item_Outlet_Sales',ax=ax[i])
            temp.set(xlabel=f"Outlet_Identifier", ylabel='Item_Outlet_Sales', title=f"Outlet Sales (K)")
            temp.set_xticklabels(temp.get_xticklabels(), rotation=30, fontsize=10)
            i += 1
            temp = sns.barplot(data=df_final,x='Outlet_Identifier',y='Item_Identifier',ax=ax[i])
            temp.set(xlabel=f"Outlet_Identifier", ylabel='Item_Identifier', title=f"# of Items")
            temp.set_xticklabels(temp.get_xticklabels(), rotation=30, fontsize=10)
            i += 1
            df.loc[getattr(df, 'Outlet_Identifier') == 'OUT010', 'Outlet_Size'] = 'Small'
            df.loc[getattr(df, 'Outlet_Identifier') == 'OUT017', 'Outlet_Size'] = 'High'
            df.loc[getattr(df, 'Outlet_Identifier') == 'OUT045', 'Outlet_Size'] = 'Medium'

    plt.tight_layout()
    plt.savefig(os.path.join(output, fname))
    return df

# categorical_conversion()
def categorical_conversion(df=None, categorical_feature=None
                           ,encoding_method={'Item_Identifier': 'one-hot',
                                             'Item_Fat_Content': 'one-hot',
                                             'Item_Type': 'one-hot',
                                             'Outlet_Identifier': 'freq',
                                             'Outlet_Establishment_Year': 'freq',
                                             'Outlet_Size': 'ordinal',
                                             'Outlet_Location_Type': 'ordinal',
                                             'Outlet_Type': 'ordinal'}, 
                           output=r'../public/output', fname='categorical_conversion.png',
                           subplots=[2,4], figsize=(40,20)):
    """ Convert categorical features into numeric

    Args:
        df: DataFrame
        categorical_feature: set, categorical feature names
        encoding_method, dict, encoding method by feature
        output: path 
        fname: output file name

    Returns:
        resulting dataframe
    """
    sns.set()
    fig, axes = plt.subplots(subplots[0], subplots[1], figsize=figsize)
    ax, i = axes.flatten(), 0

    for cat in categorical_feature:
        # Item_Identifier
        if cat == 'Item_Identifier':
            df_0 = df[['Item_Identifier','Item_Outlet_Sales']].groupby(by='Item_Identifier').count().reset_index()
            df_0 = df.merge(df_0, how='left', on='Item_Identifier')[['Item_Identifier', 'Item_Outlet_Sales_y', 'Item_Outlet_Sales_x']]\
                .rename(columns={'Item_Outlet_Sales_y': 'Item_Freq', 'Item_Outlet_Sales_x': 'Item_Outlet_Sales'})
            temp = sns.violinplot(data=df_0, x='Item_Freq', y='Item_Outlet_Sales', ax=ax[i])
            temp.set(xlabel=f"Item Identifier Frequency", ylabel=f"Dist of Item_Outlet_Sales", title=f"Volinplot of Item_Identifier Frequency")
            ohc = ce.OneHotEncoder(cols='Item_Freq', return_df=True, use_cat_names=True)
            df_0 = (ohc.fit_transform(df_0)).drop(columns = ['Item_Outlet_Sales'])
            df = df.merge(df_0,how='left', on='Item_Identifier').drop(columns=['Item_Freq_10.0']).rename(\
                columns={'Item_Freq_1.0': 'Item_Identifier_1', 'Item_Freq_2.0': 'Item_Identifier_2', 'Item_Freq_3.0': 'Item_Identifier_3',
                         'Item_Freq_4.0': 'Item_Identifier_4', 'Item_Freq_5.0': 'Item_Identifier_5', 'Item_Freq_6.0': 'Item_Identifier_6',
                         'Item_Freq_7.0': 'Item_Identifier_7', 'Item_Freq_8.0': 'Item_Identifier_8', 'Item_Freq_9.0': 'Item_Identifier_9'})
            i += 1
        # Item_Fat_Content
        elif cat == 'Item_Fat_Content':
            temp = sns.violinplot(data=df, x='Item_Fat_Content', y='Item_Outlet_Sales', ax=ax[i])
            temp.set(xlabel=f"Item_Fat_Content", ylabel=f"Dist of Item_Outlet_Sales", title=f"Volinplot of Item_Fat_Content")
            ohc = ce.OneHotEncoder(cols='Item_Fat_Content', return_df=True, use_cat_names=True)
            df = (ohc.fit_transform(df)).rename(columns = {'Item_Fat_Content_Low Fat': 'Low_Fat', 'Item_Fat_Content_Regular': 'Regular'})
            i += 1
        # Item_Type
        elif cat == 'Item_Type':
            temp = sns.violinplot(data=df, x=cat, y='Item_Outlet_Sales', ax=ax[i])
            temp.set(xlabel=f"{cat}", ylabel=f"Dist of Item_Outlet_Sales", title=f"Volinplot of {cat}")
            temp.set_xticklabels(temp.get_xticklabels(), rotation=30, fontsize=10)
            ohc = ce.OneHotEncoder(cols={cat}, return_df=True, use_cat_names=True)
            df = (ohc.fit_transform(df)).rename(columns = {\
                'Item_Type_Dairy': 'Dairy', 'Item_Type_Soft Drinks': 'Soft_Drinks', 'Item_Type_Meat': 'Meat',
                'Item_Type_Fruits and Vegetables': 'Fruits_Vegetables', 'Item_Type_Household': 'Household', 'Item_Type_Baking Goods': 'Baking',
                'Item_Type_Snack Foods': 'Snack', 'Item_Type_Frozen Foods': 'Fronzen_Foods', 'Item_Type_Seafood': 'Seafood',
                'Item_Type_Breakfast': 'Breakfast', 'Item_Type_Hard Drinks': 'Hard_Drinks', 'Item_Type_Health and Hygiene': 'Health_Hygiene',
                'Item_Type_Canned': 'Canned', 'Item_Type_Breads': 'Breads', 'Item_Type_Others': 'Others', 'Item_Type_Starchy Foods': 'Starchy'})
            i += 1
        # Outlet_Identifier
        elif cat == 'Outlet_Identifier':
            df_0 = df[[cat,'Item_Outlet_Sales']].groupby(by=cat).count().reset_index()
            df_0 = df.merge(df_0, how='left', on=cat)[[cat, 'Item_Outlet_Sales_y', 'Item_Outlet_Sales_x']]\
                .rename(columns={'Item_Outlet_Sales_y': 'Outlet_Freq', 'Item_Outlet_Sales_x': 'Item_Outlet_Sales'})
            temp = sns.violinplot(data=df_0, x='Outlet_Freq', y='Item_Outlet_Sales', ax=ax[i])
            temp.set(xlabel=f"Outlet Identifier Frequency", ylabel=f"Dist of Item_Outlet_Sales", title=f"Volinplot of Outlet_Identifier Frequency")
            df = df.join(df_0, how='left', rsuffix='_x').drop(columns=['Outlet_Identifier_x','Item_Outlet_Sales_x'])
            i += 1
        # Outlet_Establishment_Year
        elif cat == 'Outlet_Establishment_Year':
            df_0 = df[[cat,'Item_Outlet_Sales']].groupby(by=cat).count().reset_index()
            df_0 = df.merge(df_0, how='left', on=cat)[[cat, 'Item_Outlet_Sales_y', 'Item_Outlet_Sales_x']]\
                .rename(columns={'Item_Outlet_Sales_y': 'Outlet_Year', 'Item_Outlet_Sales_x': 'Item_Outlet_Sales'})
            temp = sns.violinplot(data=df_0, x='Outlet_Year', y='Item_Outlet_Sales', ax=ax[i])
            temp.set(xlabel=f"Outlet Year Frequency", ylabel=f"Dist of Item_Outlet_Sales", title=f"Volinplot of Outlet_Year Frequency")
            df = df.join(df_0, how='left', rsuffix='_x').drop(columns=['Outlet_Establishment_Year_x','Item_Outlet_Sales_x'])
            i += 1
        # Outlet_Size
        elif cat == 'Outlet_Size':
            temp = sns.violinplot(data=df, x=cat, y='Item_Outlet_Sales', ax=ax[i])
            temp.set(xlabel=f"{cat}", ylabel=f"Dist of Item_Outlet_Sales", title=f"Volinplot of {cat}") 
            df = df.replace({cat:{'Small': 1, 'Medium': 2, 'High': 3}})
            i += 1
        # Outlet_Location_Type
        elif cat == 'Outlet_Location_Type':
            temp = sns.violinplot(data=df, x=cat, y='Item_Outlet_Sales', ax=ax[i])
            temp.set(xlabel=f"{cat}", ylabel=f"Dist of Item_Outlet_Sales", title=f"Volinplot of {cat}") 
            df = df.replace({cat:{'Tier 1': 1, 'Tier 2': 2, 'Tier 3': 3}})
            i += 1
        # Outlet_Type
        elif cat == 'Outlet_Type':
            temp = sns.violinplot(data=df, x=cat, y='Item_Outlet_Sales', ax=ax[i])
            temp.set(xlabel=f"{cat}", ylabel=f"Dist of Item_Outlet_Sales", title=f"Volinplot of {cat}") 
            temp.set_xticklabels(temp.get_xticklabels(), rotation=30, fontsize=10)
            df = df.replace({cat:{'Grocery Store': 1, 'Supermarket Type1': 2, 'Supermarket Type2': 3, 'Supermarket Type3': 4}})
            i += 1

    plt.tight_layout()
    plt.savefig(os.path.join(output, fname))
    df = df.drop_duplicates()
    df.drop(columns=['Item_Identifier', 'Outlet_Identifier', 'Outlet_Establishment_Year'], inplace=True)
    # df.to_csv(os.path.join(output, 'data_convert.csv'), index=False)
    return df

# numeric_conversion()
def numeric_conversion(df=None, numeric_feature=None, target=None,
                       output=r'../public/output', fname='corr_heatmap.png', figsize=(10,10)):
    """ Scaling

    Args:
        df: DataFrame
        numeric_feature: set, numeric feature names
        encoding_method, dict, encoding method by feature
        output: path 
        fname: output file name

    Returns:
        resulting dataframe
    """
    corr = df[list(numeric_feature)].corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, annot=True)
    plt.savefig(os.path.join(output, fname))

    for col in numeric_feature:
        if col in ('Item_Weight', 'Item_MRP', 'Outlet_Freq'):
            scaler = preprocessing.MinMaxScaler()
            df[col] = scaler.fit_transform(pd.DataFrame(df[col]))
        elif col in ('Outlet_Year', 'Item_Visibility'):
            scaler = preprocessing.StandardScaler()
            df[col] = scaler.fit_transform(pd.DataFrame(df[col]))

    return df

# data_split()
def data_split(df=None, label=None, train_size=0.8, random_state=42, tensor=False):
    """ Split dataset into training, validation & testing
    
    Args:
        df: DataFrame
        label: str, label column name
        validation: boolean, True if a validation set is needed, otherwise False
        train_size: float, size of training dataset, <= 1
        random_state: int, random state, default value as 42
        tensor: boolean, True if need to convert to Tensor, otherwise False

    Returns:
        DataFrames, split
    """
    if tensor == False:
        x_train, x_test, y_train, y_test = train_test_split(df.iloc[:,df.columns != label], df.iloc[:,df.columns == label], 
                                                            test_size=(1-train_size), random_state=random_state)
        return x_train, x_test, y_train, y_test
    elif tensor == True:
        x_train, x_test, y_train, y_test = train_test_split(df.iloc[:,df.columns != label], df.iloc[:,df.columns == label], 
                                                            test_size=(1-train_size), random_state=random_state)
        X_train = torch.Tensor(x_train.values)
        X_test = torch.Tensor(x_test.values)
        Y_train = torch.Tensor(y_train.values)
        Y_test = torch.Tensor(y_test.values)
        return X_train, X_test, Y_train, Y_test
    
# point_eval_metric()
def point_eval_metric(y_true=None, y_pred=None, model=None):
    """ Given y_true and y_pred, generate evaluation metrics: R^2, EVS, MSE, RMSE, MAE

    Args:
        model: str
        y_true: true target values
        y_pred: predicted target values
    
    Returns:
        DataFrame with info: 
            - model, r2_score, exp_var_score, mse, rmse, mean_ae, median_ae
    """
    res = {'Model': [model],
           'R2_Score': [format(r2_score(y_true, y_pred), '.2f')],
           'Exp_Var_Score': [format(explained_variance_score(y_true, y_pred), '.2f')],
           'MSE': [format(mean_squared_error(y_true, y_pred, squared=True), '.2f')],
           'RMSE': [format(mean_squared_error(y_true, y_pred, squared=False), '.2f')],
           'Mean_Abs_Err': [format(mean_absolute_error(y_true, y_pred), '.2f')],
           'Median_Abs_Err': [format(median_absolute_error(y_true, y_pred), '.2f')]
          }

    return pd.DataFrame.from_dict(res)