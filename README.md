# MDME
A repository containing implementation of the MDME module from the "Billion-user Customer Lifetime Value Prediction: An Industrial-scale Solution from Kuaishou" paper.

## Description
[CIKM '22 "Billion-user Customer Lifetime Value Prediction: An Industrial-scale Solution from Kuaishou"](https://dl.acm.org/doi/10.1145/3511808.3557152) is a paper for customer lifetime value (LTV) prediction by Kungpeng Li, Guangcui Shao, Naijun Yang, Xiao Fang and Yang Song published at the Proceedings of the 31st ACM International Conference on Information & Knowledge Management in 2022.

I was interested in implementing part of the Order Dependency Monotonic Network (ODMN) described in the paper. The block I was interested in is callsed Milto Distribution Multi Experts (MDME) module which models LTV values by dividing them into multiple distributions and modeling each one separately. I test my code on the [Kaggle's Acquire Valued Shoppers Challenge](https://www.kaggle.com/c/acquire-valued-shoppers-challenge) dataset, which contains consumers' transaction records of purchases.

## Code
The code in this repository is using multiple sources, and I try to list them here, but also in the corresponding scripts. The kaggle competition originally wasn't meant for LTV predictions, so data needs to be preprocessed first. Data preprocessing part is taken from the "A Deep Probabilistic Model for Consumer Lifetime Value Prediction (2019)"[https://arxiv.org/abs/1912.07753] paper, and in particular the corresponding [github](https://github.com/google/lifetime_value), and the [notebook demonstrating the data preprocessing step](https://github.com/google/lifetime_value/blob/master/notebooks/kaggle_acquire_valued_shoppers_challenge/preprocess_data.ipynb). My code assumes that you've already downloaded data from kaggle, unzipped it and put it into the "data" folder. 

Metrics calculation like normalized Gini coefficient, splitting the data into train/test parts is also based on the same Google's `lifetime_value` repository.

**MDME** module uses Ordinal Regression to help learning as distribution and bucket labels are ordered. A higher label value means a higher LTV. There are multiple ways to implement an Ordinal Regression through Deep Learning. For example, representing a problem as a binary classification (a multi-label classification). However, I am using an implementation that appeared in the [Rank consistent ordinal regression for neural networks with application to age estimation, 2020](https://arxiv.org/abs/1901.07884). It is [implemented in PyTorch](https://github.com/Raschka-research-group/coral-cnn), but I take the [Tensorflow implementation from here](https://github.com/ck37/coral-ordinal).


