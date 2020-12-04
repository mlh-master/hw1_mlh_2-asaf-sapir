# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:14:23 2019

@author: smorandv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rm_ext_and_nan(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A dictionary of clean CTG called c_ctg
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    c_ctg = {key: pd.to_numeric(CTG_features[key], errors='coerce').dropna() for key in CTG_features.keys() if
             key != extra_feature}
    # --------------------------------------------------------------------------
    return c_ctg


def nan2num_samp(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A pandas dataframe of the dictionary c_cdf containing the "clean" features
    """
    c_cdf = {}
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    for key in CTG_features.keys():
        if key != extra_feature:
            none_nan = pd.to_numeric(CTG_features[key], errors='coerce', downcast='float').dropna()
            freq = dict()
            for element in none_nan:
                if element in freq:
                    freq[element] += 1
                else:
                    freq[element] = 1

            for number in freq.keys():
                freq[number] = freq[number] / len(none_nan)
            elements = list(freq.keys())
            probs = list(freq.values())

            c_cdf[key] = CTG_features[key].copy()
            for counter, value in enumerate(pd.to_numeric(c_cdf[key], errors='coerce')):
                if np.isnan(value):
                    c_cdf[key][counter + 1] = np.random.choice(elements, 1, probs)[0]
            c_cdf[key] = c_cdf[key].astype(float)
    # -------------------------------------------------------------------------
    return pd.DataFrame(c_cdf)


def sum_stat(c_feat):
    """

    :param c_feat: Output of nan2num_cdf
    :return: Summary statistics as a dicionary of dictionaries (called d_summary) as explained in the notebook
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    d_summary = {}
    columns_list = list(c_feat.columns)
    for column in columns_list:
        d_summary[column] = dict()
        d_summary[column]["min"] = min(c_feat[column])
        d_summary[column]["Q1"] = np.percentile(c_feat[column], 25)
        d_summary[column]["median"] = np.percentile(c_feat[column], 50)
        d_summary[column]["Q3"] = np.percentile(c_feat[column], 75)
        d_summary[column]["max"] = max(c_feat[column])
    # -------------------------------------------------------------------------
    return d_summary


def rm_outlier(c_feat, d_summary):
    """

    :param c_feat: Output of nan2num_cdf
    :param d_summary: Output of sum_stat
    :return: Dataframe of the dictionary c_no_outlier containing the feature with the outliers removed
    """
    c_no_outlier = {}
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    columns_list = list(c_feat.columns)
    for column in columns_list:
        Q1 = d_summary[column]["Q1"]
        Q3 = d_summary[column]["Q3"]
        IQR = Q3 - Q1
        Med = d_summary[column]["median"]

        values = c_feat[column].copy()
        for idx, x in enumerate(values):
            if (x > Q3 + 1.5 * IQR) or (x < Q1 - 1.5 * IQR):
                values[idx + 1] = np.NaN
        c_no_outlier[column] = values
    # -------------------------------------------------------------------------
    return pd.DataFrame(c_no_outlier)


def phys_prior(c_cdf, feature, thresh):
    """

    :param c_cdf: Output of nan2num_cdf
    :param feature: A string of your selected feature
    :param thresh: A numeric value of threshold
    :return: An array of the "filtered" feature called filt_feature
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:-----------------------------
    filt_feature = c_cdf[feature].copy()
    for idx, x in enumerate(filt_feature):
        if x > thresh:
            filt_feature[idx + 1] = None
    filt_feature = filt_feature.dropna()
    # -------------------------------------------------------------------------
    return filt_feature


def norm_standard(CTG_features, selected_feat=('LB', 'ASTV'), mode='none', flag=False):
    """

    :param CTG_features: Pandas series of CTG features
    :param selected_feat: A two elements tuple of strings of the features for comparison
    :param mode: A string determining the mode according to the notebook
    :param flag: A boolean determining whether or not plot a histogram
    :return: Dataframe of the normalized/standardazied features called nsd_res
    """
    x, y = selected_feat
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    # Case of no normalized/standardazied features
    if mode == 'none':
        nsd_res = CTG_features.copy()

        if flag:
            # Histograms
            xlbl = ['beats/min', '%']
            axarr = nsd_res.hist(column=[x, y], bins=100, layout=(1, 2))
            for i, ax in enumerate(axarr.flatten()):
                ax.set_xlabel(xlbl[i])
                ax.set_ylabel("Count")
            plt.suptitle('No Normalization/Standardization', fontsize=12)
            plt.show()

    # Case of Standard scaling
    if mode == 'standard':
        nsd_res = CTG_features.copy()  # copy the dataframe
        # apply standard scaling
        for column in nsd_res.columns:
            nsd_res[column] = (nsd_res[column] - nsd_res[column].mean()) / nsd_res[column].std()

        if flag:
            # Histograms
            axarr = nsd_res.hist(column=[x, y], bins=100, layout=(1, 2))
            for i, ax in enumerate(axarr.flatten()):
                ax.set_ylabel("Count")
            plt.suptitle('Standard scaling', fontsize=12)
            plt.show()

    # Case of Min-Max scaling
    if mode == 'MinMax':
        nsd_res = CTG_features.copy()  # copy the dataframe
        # apply min-max scaling
        for column in nsd_res.columns:
            nsd_res[column] = (nsd_res[column] - nsd_res[column].min()) / (
                        nsd_res[column].max() - nsd_res[column].min())

        if flag:
            # Histograms
            axarr = nsd_res.hist(column=[x, y], bins=100, layout=(1, 2))
            for i, ax in enumerate(axarr.flatten()):
                ax.set_ylabel("Count")
            plt.suptitle('Min-Max scaling', fontsize=12)
            plt.show()

    # Case of Mean scaling
    if mode == 'mean':
        nsd_res = CTG_features.copy()  # copy the dataframe
        # apply min-max scaling
        for column in nsd_res.columns:
            nsd_res[column] = (nsd_res[column] - nsd_res[column].mean()) / (
                        nsd_res[column].max() - nsd_res[column].min())

        if flag:
            # Histograms
            axarr = nsd_res.hist(column=[x, y], bins=100, layout=(1, 2))
            for i, ax in enumerate(axarr.flatten()):
                ax.set_ylabel("Count")
            plt.suptitle('Mean scaling', fontsize=12)
            plt.show()
    # -------------------------------------------------------------------------
    return pd.DataFrame(nsd_res)
