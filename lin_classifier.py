import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold as SKFold
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
import pandas as pd
import scipy.stats as stats
from clean_data import norm_standard as nsd


def pred_log(logreg, X_train, y_train, X_test, flag=False):
    """

    :param logreg: An object of the class LogisticRegression
    :param X_train: Training set samples
    :param y_train: Training set labels 
    :param X_test: Testing set samples
    :param flag: A boolean determining whether to return the predicted probabilities of the classes (relevant after Q11)
    :return: A two elements tuple containing the predictions and the weightning matrix
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:-----------------------------
    # Logistic Regression Model Fitting
    logreg.fit(X_train, y_train)
    if flag:
        y_pred_log = logreg.predict_proba(X_test)
    else:
        y_pred_log = logreg.predict(X_test)
    w_log = logreg.coef_
    # -------------------------------------------------------------------------
    return y_pred_log, w_log


def w_no_p_table(w, features):
    x = np.arange(len(features))
    width = 0.5  # the width of the bars
    mode_name = ['Normal', 'Suspect', 'Pathology']
    fig, axs = plt.subplots(figsize=(20, 10), nrows=3)
    for idx, ax in enumerate(axs):
        ax.bar(x, w[idx, :], width)
        ax.set(xticks=x, xticklabels=features, ylabel='w', title=mode_name[idx])
    fig.tight_layout()
    plt.show()


def w_all_tbl(w2, w1, orig_feat):
    idx_l2 = np.argsort(-w2, axis=1)
    w2_sort = -np.sort(-w2, axis=1)
    w1_sort = np.zeros(w2_sort.shape)
    mode_name = ['Normal', 'Suspect', 'Pathology']
    lbl = ['L2', 'L1']
    col = ['orange', 'green']
    feature_dict = {}
    for i in range(w2_sort.shape[0]):
        w1_sort[i, :] = w1[i, idx_l2[i, :]]
        feature_dict[mode_name[i]] = [orig_feat[x] for x in idx_l2[i, :]]
    width = 0.4
    w_tot = [w2_sort, w1_sort]
    fig, axs = plt.subplots(figsize=(20, 10), nrows=3)
    x_orig = np.arange(len(orig_feat))
    x = np.arange(len(orig_feat)) - width/2
    for idx_w, w in enumerate(w_tot):
        for idx_ax, ax in enumerate(axs):
            ax.bar(x, w[idx_ax, :], width, label=lbl[idx_w], color=col[idx_w])
            ax.set(xticks=x_orig, xticklabels=feature_dict[mode_name[idx_ax]], ylabel='w', title=mode_name[idx_ax])
            ax.legend()
        x += width
    fig.tight_layout()
    plt.show()


def cv_kfold(X, y, C, penalty, K, mode):
    """
    
    :param X: Training set samples
    :param y: Training set labels 
    :param C: A list of regularization parameters
    :param penalty: A list of types of norm
    :param K: Number of folds
    :param mode: Mode of normalization (parameter of norm_standard function in clean_data module)
    :return: A dictionary as explained in the notebook
    """
    kf = SKFold(n_splits=K)
    validation_dict = []
    for c in C:
        for p in penalty:
            logreg = LogisticRegression(solver='saga', penalty=p, C=c, max_iter=10000, multi_class='ovr')
            loss_val_vec = np.zeros(K)
            k = 0
            for train_idx, val_idx in kf.split(X, y):
                x_train, x_val = X.iloc[train_idx], X.iloc[val_idx]
        # ------------------ IMPLEMENT YOUR CODE HERE:-----------------------------
                x_train_norm = nsd(x_train, selected_feat=('LB', 'ASTV'), mode=mode, flag=False)
                x_val_norm = nsd(x_val, selected_feat=('LB', 'ASTV'), mode=mode, flag=False)
                y_val_norm, w_norm_std = pred_log(logreg, x_train_norm, y[train_idx], x_val_norm, flag=True)
                loss_val_vec[k] = log_loss(y[val_idx], y_val_norm)

            validation_dict.append({'C': c, 'penalty': p, 'mu': np.mean(loss_val_vec), 'sigma': np.std(loss_val_vec)})
        # --------------------------------------------------------------------------
    return validation_dict


def odds_ratio(w, X, selected_feat='LB'):
    """

    :param w: the learned weights of the non normalized/standardized data
    :param X: the set of the relevant features-patients data
    :param selected_feat: the current feature
    :return: odds: median odds of all patients for the selected feature and label
             odds_ratio: the odds ratio of the selected feature and label
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:-----------------------------
    # Calculate odds ratio
    all_odds_ratio = np.exp(w)
    index = X.columns.get_loc(selected_feat)
    odd_ratio = all_odds_ratio[0, index]

    # Calculate odds of 'Normal'
    multi = w.dot(np.transpose(np.array(X)))[0, :]  # Keep only the 'Normal' row
    odds_list = np.zeros(len(multi))
    for idx, elem in enumerate(multi):
        prob = np.exp(elem)/(1 + np.exp(elem))
        odd = prob/(1-prob)
        odds_list[idx] = odd
    odds = np.median(odds_list)
    # --------------------------------------------------------------------------
    return odds, odd_ratio