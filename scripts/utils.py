""" Collection of functions to help process the data for LTV prediction

Some functions are taken from the  "Google Life-Time Value prediction" repo. This repo is an accompanying code for the following paper:
Wang, Xiaojing, Liu, Tianqi, and Miao, Jingang. (2019). A Deep Probabilistic Model for Customer Lifetime Value Prediction. arXiv:1912.07753.

And here is the link to the repo:
https://github.com/google/lifetime_value
"""


import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from loss import _label_to_levels


CATEGORICAL_FEATURES = ['chain', 'dept', 'category', 'brand', 'productmeasure']
NUMERIC_FEATURES = ['log_calibration_value']
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def ordinal_to_binary(y_int: pd.Series):
    """
    Convert integer ordinal labels into binary ordinal format.
    
    Args:
        y_int: pd.Series of shape (N,) containing integer ordinal labels.
        num_bins: Total number of ordinal bins.
    
    Returns:
        Tensor of shape (N, num_bins) with binary ordinal encoding where num_bins is the number of unique labels
    """
    N = tf.shape(y_int)[0]
    num_bins = int(y_pred.get_shape().as_list()[1]) + 1
    y_int = tf.convert_to_tensor(y_int, np.int64)
    range_tensor = tf.range(num_bins, dtype=tf.int64)  # [0, 1, 2, ..., num_bins-1]
    
    # Expand dimensions for broadcasting
    y_int = tf.expand_dims(y_int, axis=-1)  # Shape: (N, 1)
    range_tensor = tf.expand_dims(range_tensor, axis=0)  # Shape: (1, num_bins)
    
    # Compare: If ordinal index >= bin index → 1, otherwise 0
    binary_labels = tf.cast(y_int >= range_tensor, dtype=tf.float32)
    
    return binary_labels



def dnn_split(df):
    for key in CATEGORICAL_FEATURES:
        encoder = LabelEncoder()
        df[key] = encoder.fit_transform(df[key])
    
    df['unique_distribution_buckets'] = df['label_distribution'].astype('str') + df['label_bucket'].astype('str')
    categories = df['unique_distribution_buckets'].unique()
    y0 = df['calibration_value'].values
    df_train, df_eval, y0_train, y0_eval = model_selection.train_test_split(df, y0, test_size=0.2, random_state=123)
    
    

    for bucket in categories:
        values = df_train.loc[df_train['unique_distribution_buckets'] == bucket, 'label'].values.reshape(-1, 1)
        values_eval = df_eval.loc[df_eval['unique_distribution_buckets'] == bucket, 'label'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(values)
        df_train.loc[df_train['unique_distribution_buckets'] == bucket, 'label_minmax'] = scaled_values
        df_eval.loc[df_eval['unique_distribution_buckets'] == bucket, 'label_minmax'] = scaler.transform(values_eval)

    def feature_dict(df):
        features = {k: v.values for k, v in dict(df[CATEGORICAL_FEATURES]).items()}
        features['numeric'] = df[NUMERIC_FEATURES].values
        return features

    y_train_dict = {
        'label_distribution': df_train['label_distribution'],
        'label_ordinality': df_train['label_distribution'],
        'bucket_distribution': df_train['label_bucket'],
        'bucket_ordinality': df_train['label_bucket'],
        'label': df_train['label'].values,
        'label_minmax': df_train['label_minmax'].values
        }

    y_eval_dict = {
        'label_distribution': df_eval['label_distribution'],
        'label_ordinality': df_eval['label_distribution'],
        'bucket_distribution': df_eval['label_bucket'],
        'bucket_ordinality': df_eval['label_bucket'],
        'label': df_eval['label'].values,
        'label_minmax': df_eval['label_minmax'].values
        }


    x_train, y_train = feature_dict(df_train), y_train_dict
    x_eval, y_eval = feature_dict(df_eval), y_eval_dict

    return x_train, x_eval, y_train, y_eval, y0_eval
