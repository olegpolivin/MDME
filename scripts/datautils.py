"""
Collections of functions to load data for the Kaggle Challenge Dataset
https://www.kaggle.com/c/acquire-valued-shoppers-challenge

These functions are taken from the "Google Life-Time Value prediction" repo. This repo is an accompanying code for the following paper:
Wang, Xiaojing, Liu, Tianqi, and Miao, Jingang. (2019). A Deep Probabilistic Model for Customer Lifetime Value Prediction. arXiv:1912.07753.

And here is the link to the repo:
https://github.com/google/lifetime_value
"""

import os
import numpy as np
import pandas as pd


CATEGORICAL_FEATURES = ['chain', 'dept', 'category', 'brand', 'productmeasure']
NUMERIC_FEATURES = ['log_calibration_value']
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def load_transaction_data(company):
    all_data_filename = '../data/kaggle-challenge/transactions.csv.gz'
    one_company_data_filename = ('../data/kaggle-challenge/transactions_company_{}.csv'.format(company))
    if os.path.isfile(one_company_data_filename):
        df = pd.read_csv(one_company_data_filename)
    else:
        df = pd.read_csv(all_data_filename, compression='gzip', header=0)
        df = df.query("company=={}".format(company))
        df.to_csv(one_company_data_filename, index=None)
    return df


def preprocess(df):
    df = df.query('purchaseamount>0')
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['start_date'] = df.groupby('id')['date'].transform('min')

    # Compute calibration values
    calibration_value = (
        df.query('date==start_date').groupby('id')
        ['purchaseamount'].sum().reset_index())
    calibration_value.columns = ['id', 'calibration_value']

    # Compute holdout values
    one_year_holdout_window_mask = (
        (df['date'] > df['start_date']) &
        (df['date'] <= df['start_date'] + np.timedelta64(365, 'D')))
    holdout_value = (
        df[one_year_holdout_window_mask].groupby('id')
        ['purchaseamount'].sum().reset_index())
    holdout_value.columns = ['id', 'holdout_value']

    # Compute calibration attributes
    calibration_attributes = (
        df.query('date==start_date').sort_values(
            'purchaseamount', ascending=False).groupby('id')[[
            'chain', 'dept', 'category', 'brand', 'productmeasure'
        ]].first().reset_index())

    # Merge dataframes
    customer_level_data = (
        calibration_value.merge(calibration_attributes, how='left',
                                on='id').merge(holdout_value, how='left', on='id'))

    customer_level_data['holdout_value'] = (customer_level_data['holdout_value'].fillna(0.))
    customer_level_data[CATEGORICAL_FEATURES] = (customer_level_data[CATEGORICAL_FEATURES].fillna('UNKNOWN'))
    
    # Specify data types
    customer_level_data['log_calibration_value'] = (np.log(customer_level_data['calibration_value']).astype('float32'))
    customer_level_data['chain'] = (customer_level_data['chain'].astype('category'))
    customer_level_data['dept'] = (customer_level_data['dept'].astype('category'))
    customer_level_data['brand'] = (customer_level_data['brand'].astype('category'))
    customer_level_data['category'] = (customer_level_data['category'].astype('category'))
    customer_level_data['label'] = (customer_level_data['holdout_value'].astype('float32'))
    return customer_level_data


def load_customer_level_csv(company):
    customer_level_data_file = (
        '../data/kaggle-challenge/customer_level_data_company_{}.csv'.format(company))
    customer_level_data = preprocess(load_transaction_data(company))
    for cat_col in CATEGORICAL_FEATURES:
        customer_level_data[cat_col] = (customer_level_data[cat_col].astype('category'))

    for num_col in ['log_calibration_value', 'calibration_value', 'holdout_value']:
        customer_level_data[num_col] = (customer_level_data[num_col].astype('float32'))
    return customer_level_data
