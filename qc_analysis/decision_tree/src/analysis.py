#!/usr/bin/env python
# coding: utf-8
############################################################################################################
# Authors:
#   Anirudh Bhashyam, Uni Heidelberg, anirudh.bhashyam@stud.uni-heidelberg.de   (Current Maintainer)
# Date: 04/2022
# License: Contact authors
###
# Algorithms for:
#   Decision tree evaluation by medaka_bpm.
###
############################################################################################################
import os
import time
import joblib
from typing import Iterable, Union, Tuple, List

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

from imblearn.over_sampling import SMOTE

#################### -- GLOBALS -- ####################
TEST_SET_SIZE = 0.3

PLOT_SAVE_DIR = "figs"
DATA_SAVE_DIR = "thresholds"
TREE_SAVE_DIR = os.path.join(__file__, *(4 * [os.pardir]), "data")

LABELS = "error"

QC_FEATURES = ["HROI Change Intensity", "Harmonic Intensity", "SNR", "Signal intensity", "Signal regional prominence", "Intensity/Harmonic Intensity (top 5 %)", "SNR Top 5%", "Signal Intensity Top 5%"]
#################### -- GLOBALS -- ####################

def plot_qc_params(
    data: pd.DataFrame,
    save_name: str, 
    out_dir: str,
    limits: Union[dict, None] = None,
    figsize: tuple = (10, 30), 
    save_q: bool = True
) -> None:

    fig, ax = plt.subplots(nrows = len(QC_FEATURES), 
                        figsize = figsize,
                        gridspec_kw = dict(left = 0.01, right = 0.9,
                                            bottom = 0.0001, top = 0.9)) 

    for i, feature in enumerate(QC_FEATURES):
        ax[i].set(ylabel = feature) 
        sns.scatterplot(
            data = data,
            x = LABELS, 
            y = feature,
            s = 4,
            ax = ax[i]
        )
        if limits:
            if feature in limits:
                # Plot the threshold line for each feature.
                for threshold in limits[feature]:
                    ax[i].axhline(y = threshold, xmin = 0, xmax = 200, c = "red", lw = 0.7, label = threshold)    
                    ax[i].legend()
                    
    if save_q:
        fig.savefig(os.path.join(out_dir, ".".join([save_name, "png"])), 
                    dpi = 80,
                    bbox_inches = "tight")
        
    plt.close()

def convert_error_cat(actual: Iterable, desired: Iterable, threshold: float) -> list:
    # 1 -> ERROR and 0 -> NO ERROR.
    return [1 if abs(a - d) >= threshold else 0 for a, d in zip(actual, desired)]

def process_data(raw_data: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, np.array]:
    # Columns to drop will also drop prexisting error columns.
    actual = "Heartrate (BPM)"
    desired = "ground truth"
    
    data = raw_data[QC_FEATURES + [actual, desired]].copy()
    
    data[LABELS] = convert_error_cat(data[actual], data[desired], threshold)
    
    data = data.drop(columns = [actual, desired])
    
    scaler = MinMaxScaler()
    scaler.fit(data[QC_FEATURES])
    data[QC_FEATURES] = scaler.transform(data[QC_FEATURES])
    
    return data, scaler.scale_

def decision_tree(data: pd.DataFrame, out_dir: str) -> Tuple[sklearn.tree.DecisionTreeClassifier, dict]:
    Y = data.pop(LABELS)

    oversample = SMOTE()
    over_X, over_Y = oversample.fit_resample(data, Y)
    over_X_train, over_X_test, over_Y_train, over_Y_test = train_test_split(over_X, over_Y, test_size=0.1, stratify = over_Y)

    # clf = DecisionTreeClassifier()
    clf = RandomForestClassifier(n_estimators = 150, random_state = 104729, class_weight = "balanced")
    
    # K-Fold cross validation (with stratification).
    folds = 6
    cv = StratifiedKFold(n_splits = folds, shuffle = True, random_state = 104729)

    # clf = GridSearchCV(
    #     estimator = clf, 
    #     param_grid = {
    #         "max_leaf_nodes": [2, 5, 10, 20],
    #         "max_features": list(range(1, len(QC_FEATURES))),
    #         "min_samples_leaf": list(range(1, 11)),
    #     },
    #     cv = cv,
    #     scoring = "accuracy",
    #     n_jobs = 4,
    # )

    scores = cross_validate(
        estimator = clf,
        X = over_X,
        y = over_Y,
        cv = cv,
        scoring = ("accuracy", "f1", "recall", "precision"),
        n_jobs = 4
    )

    X_train, X_test, Y_train, Y_test = train_test_split(
        data, 
        Y, 
        test_size = TEST_SET_SIZE,
        stratify = Y, 
        random_state = 104729
    )

    clf.fit(over_X_train, over_Y_train)

    classifier_results = {
        "cv": cv.__class__.__name__,
        "k folds": folds,
        "CV accuracy": scores["test_accuracy"].mean(),
        "CV f1": scores["test_f1"].mean(),
        "CV recall": scores["test_recall"].mean(),
        "CV precision": scores["test_precision"].mean(),
        "Train Score": clf.score(X_train, Y_train),
        "Test Score": clf.score(X_test, Y_test),
    }

    plot_confusion_matrix(
        clf, 
        X_test, 
        Y_test, 
        "cfm", 
        out_dir
    )

    return clf, {k: [v] for k, v in classifier_results.items() if not isinstance(v, list)}

def plot_confusion_matrix(
    clf,
    X_test: pd.DataFrame,
    Y_test: pd.DataFrame,
    save_name: str,
    out_dir: str,
    figsize: tuple = (10, 10),
    save_q: bool = True
) -> None:

    fig, ax = plt.subplots(1, 1, figsize = figsize)
    cfm = sklearn.metrics.confusion_matrix(Y_test, clf.predict(X_test).reshape(-1, 1))
    sns.heatmap(
        cfm, 
        annot = True, 
        fmt = "d", 
        cmap = "Blues", 
        ax = ax,
        annot_kws = {"ha": "center", "va": "center"}
    )

    if save_q:
        fig.savefig(
            os.path.join(out_dir, f"{save_name}.png"), 
            dpi = 80,
            bbox_inches = "tight"
        )

    plt.close()
    
def plot_decision_tree(
    tree: sklearn.tree.DecisionTreeClassifier,
    save_name: str,
    out_dir: str,
    feature_names: Iterable[str],
    class_names: Iterable[str] = ["no_error", "error"],
    figsize: tuple = (40, 30),
    save_q = True
) -> None:

    fig = plt.figure(figsize = figsize)
    _ = sklearn.tree.plot_tree(tree,
                           feature_names = feature_names,
                           class_names = class_names,
                           fontsize = 14,
                           filled = True)
    if save_q:
        fig.savefig(os.path.join(out_dir, ".".join(["decision_tree", "png"])), dpi = 80, bbox_inches = "tight")
        
    
def get_thresholds(
    unscaled_data: pd.DataFrame,
    train_data_features: Iterable, 
    classifier: sklearn.tree.DecisionTreeClassifier
) -> dict:

    n_nodes = classifier.tree_.node_count
    feature = classifier.tree_.feature
    threshold = classifier.tree_.threshold
    limits = {}

    for i in range(n_nodes):
        if feature[i] != -2:
            curr_col = unscaled_data[train_data_features[feature[i]]]
            # Get the threshold for the current feature by reversing the scaling.
            actual_threshold = threshold[i] * (curr_col.max() - curr_col.min()) + curr_col.min() 
            if train_data_features[feature[i]] not in limits:
                limits[train_data_features[feature[i]]] = [actual_threshold]
            else:
                limits[train_data_features[feature[i]]].append(actual_threshold)
            
    # Clean limits.
    for _, v in limits.items():
        while (-2 in v):
            v.remove(-2)
            
    return limits

def process_limits(qc_thresholds: dict) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(qc_thresholds, orient = "index")
    df["qc_mean"] = df.mean(axis = 1, skipna = True)
    df["qc_max"] = df.max(axis = 1, skipna = True)
    return df 

def write_results(
    raw_data: pd.DataFrame, 
    data: pd.DataFrame, 
    classifier: sklearn.tree.DecisionTreeClassifier, 
    classifier_results: dict, 
    #   limits: dict, 
    out_dir: str
) -> None:

    results_dir = os.path.join(out_dir, "qc_analysis_results_training")
    
    if os.path.exists(results_dir):
        print("Results directory already exists. Overwriting.")
    
    os.makedirs(results_dir, exist_ok = True)
    
    plots_dir = os.path.join(results_dir, PLOT_SAVE_DIR)
    data_dir = os.path.join(results_dir, DATA_SAVE_DIR)
    tree_dir = os.path.abspath(TREE_SAVE_DIR)
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        
    if not os.path.exists(tree_dir):
        os.makedirs(tree_dir)
        
    # plot_qc_params(
    #     data = raw_data, 
    #     limits = limits, 
    #     save_name = "qc_params_thresholds", 
    #     figsize = (10, 40), 
    #     out_dir = plots_dir
    # )

    # plot_decision_tree(
    #     tree = classifier, 
    #     feature_names = data.columns, 
    #     save_name = "decision_tree", 
    #     out_dir = plots_dir
    # )
    
    # threshold_data = process_limits(limits)
    # threshold_data.to_csv(os.path.join(data_dir, "qc_thresholds.csv"))
    
    pd.DataFrame(classifier_results).to_csv(os.path.join(data_dir, "classifier_results.csv"))
    
    # Write the tree to a sav file. This file needs to be accessed from medaka_bpm.
    joblib.dump(classifier, os.path.join(tree_dir, "decision_tree.sav"))

def evaluate(trained_tree: sklearn.tree.DecisionTreeClassifier, data: pd.DataFrame):
    return trained_tree.predict(data)
