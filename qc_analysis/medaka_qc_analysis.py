#!/usr/bin/env python
# coding: utf-8
############################################################################################################
# Authors:
#   Anirudh Bhashyam, Uni Heidelberg, anirudh.bhashyam@stud.uni-heidelberg.de   (Current Maintainer)
# Date: 04/2022
# License: Contact authors
###
# Algorithms for:
#   Classification evaluation by medaka_bpm to verify BPM correctness.
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

from decision_tree.src.analysis import DataHandler, RFClassifier, DTClassifier, IOHandler, Plot

from imblearn.over_sampling import SMOTE

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
    config = Config()
    config_data = next(config.load())
    args = process_args()
    data_handler = DataHandler(args.input_file)
    io_handler = IOHandler(args.out_dir)
    clf = RFClassifier(
        n_estimators = 400,
        random_state = 104729,
        folds = 10,
        scoring = ("f1", "precision", "recall", "accuracy"),
        n_jobs = 4,
        clf = RandomForestClassifier,
        cv = StratifiedKFold,
        sampling = SMOTE
    )

    data, scale = data_handler.process(args.error_threshold)
    Y = data.pop(analyse.LABELS)
    # clf = DTClassifier(
    #     max_features = "sqrt",
    #     random_state = 104729,  
    #     folds = 6,
    #     scoring = ("f1", "precision", "recall", "accuracy"),
    #     n_jobs = 4,
    #     clf = DecisionTreeClassifier, 
    #     cv = StratifiedKFold,
    # )
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
    main()