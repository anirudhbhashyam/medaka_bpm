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
import sys
import argparse

import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import  RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

import decision_tree.src.analysis as analyse

from decision_tree.src.analysis_rewrite import DataHandler, RFClassifier, DTClassifier, IOHandler, Plot

def process_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file", 
        type = str, 
        help = "Training data file (csv)"
    )

    parser.add_argument(
        "out_dir",  
        type = str,
        help = "Directory to write analysis results", 
    )

    parser.add_argument(
        "-et",
        "--error_threshold",
        nargs = "?",
        type = float,
        default = 2.0,
        help = "Error threshold (bpm) for decision tree"
    )

    return parser.parse_args()

def main() -> None:
    args = process_args()
    raw_data = analyse.pd.read_csv(args.input_file)
    data, scale = analyse.process_data(raw_data, args.error_threshold)
    classifier, classifier_results = analyse.decision_tree(data, args.out_dir)
    # limits = analyse.get_thresholds(raw_data, analyse.QC_FEATURES, classifier)
    # analyse.write_results(raw_data, data, classifier, classifier_results, limits, args.out_dir)
    analyse.write_results(raw_data, data, classifier, classifier_results, args.out_dir)

def main2() -> None:
    args = process_args()
    data_handler = DataHandler(args.input_file)
    io_handler = IOHandler(args.out_dir)
    data, scale = data_handler.process(args.error_threshold)
    Y = data.pop(analyse.LABELS)
    # clf = RFClassifier(
    #     n_estimators = 150,
    #     random_state = 104729,
    #     folds = 10,
    #     scoring = ("f1", "precision", "recall", "accuracy"),
    #     n_jobs = 4,
    #     clf = RandomForestClassifier,
    #     cv = StratifiedKFold,
    # )
    clf = DTClassifier(
        max_features = "sqrt",
        random_state = 104729,  
        folds = 6,
        scoring = ("f1", "precision", "recall", "accuracy"),
        n_jobs = 4,
        clf = DecisionTreeClassifier, 
        cv = StratifiedKFold,
    )

    clf.fit(data, Y)
    plot_cfm = Plot(
        clf = clf, 
        figsize = (10, 10), 
        save_name = "cfm.png",
    )

    plot_cfm.confusion_matrix(data, Y)
    io_handler.write_figs(plot_cfm)
    io_handler.write_results(clf)

if __name__ == "__main__":
    main2()