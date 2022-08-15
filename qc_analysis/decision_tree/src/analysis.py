#!/usr/bin/env python
# coding: utf-8
############################################################################################################
# Authors:
#   Anirudh Bhashyam, Uni Heidelberg, anirudh.bhashyam@stud.uni-heidelberg.de   (Current Maintainer)
# Date: 08/2022
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

from abc import ABC, abstractmethod

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

from sklearn.utils.validation import NotFittedError
### --------------------------------------------------------------------------------------------------------- ###
### GLOBALS
### --------------------------------------------------------------------------------------------------------- ###
TEST_SET_SIZE = 0.3

PLOT_SAVE_DIR = "figs"
DATA_SAVE_DIR = "thresholds"
TREE_SAVE_DIR = os.path.join(__file__, *(4 * [os.pardir]), "data")

LABELS = "error"

QC_FEATURES = ["HROI Change Intensity", "Harmonic Intensity", "SNR", "Signal intensity", "Signal regional prominence", "Intensity/Harmonic Intensity (top 5 %)", "SNR Top 5%", "Signal Intensity Top 5%"]
### --------------------------------------------------------------------------------------------------------- ###
### GLOBALS
### --------------------------------------------------------------------------------------------------------- ###

class DataHandler:
    def __init__(self, path: str) -> None:
        self.path = path
        self.data = self._load()

    def _load(self) -> pd.DataFrame:
        return pd.read_csv(
            self.path,
            header = 0
        )
    
    def process(self, threshold: float) -> Tuple[pd.DataFrame, np.array]:
        actual = "Heartrate (BPM)"
        desired = "ground truth"
        data = self.data[QC_FEATURES + [actual, desired]].copy()
        data[LABELS] = self.convert_error_cat(data[actual], data[desired], threshold)
        data = data.drop(columns = [actual, desired])
        scaler = MinMaxScaler()
        scaler.fit(data[QC_FEATURES])
        data[QC_FEATURES] = scaler.transform(data[QC_FEATURES])
        return data, scaler.scale_

    @staticmethod
    def convert_error_cat(actual: Iterable, desired: Iterable, threshold: float) -> list:
        return [1 if abs(a - d) >= threshold else 0 for a, d in zip(actual, desired)]

class Classifier(ABC):
    def __init__(
        self, 
        random_state: int,  
        folds: int,
        scoring: Tuple[str], 
        n_jobs: int,
        clf: sklearn.base.ClassifierMixin,
        cv: sklearn.model_selection.BaseCrossValidator, 
    ) -> None:
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.folds = folds
        self.random_state = random_state
        self.clf = clf
        self.cv = cv
        self.results = None

    @abstractmethod
    def fit(self, data: pd.DataFrame, Y: pd.DataFrame) -> None:
        pass

    def predict(self, data: pd.DataFrame) -> np.array:
        return self.clf.predict(data)

    def _save(self, out_dir: str) -> None:
        if self.results is None:
            raise NotFittedError("Classifier not fitted.")

        # Specific directory for the model.
        model_dir = os.path.abspath(TREE_SAVE_DIR)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok = True)
        joblib.dump(
            self.clf, 
            os.path.join(
                model_dir,
                f"decision_tree.sav"
            )
        )
        self.results = {k : [v] for k, v in self.results.items()}
        pd.DataFrame\
            .from_dict(
                self.results, 
                orient = "index",
                columns = ["Value"]
            )\
            .to_csv(
                os.path.join(out_dir, "classifier_results.csv")
            )

    @property
    def _cv_name(self) -> str:
        return self.cv.__class__.__name__

class RFClassifier(Classifier):
    def __init__(
        self, 
        n_estimators: int,
        random_state: int,  
        folds: int,
        scoring: Tuple[str], 
        n_jobs: int,
        clf: RandomForestClassifier, 
        cv: sklearn.model_selection.BaseCrossValidator,
        sampling = None,
    ) -> None:

        super().__init__(
            random_state = random_state,
            folds = folds,
            scoring = scoring,
            n_jobs = n_jobs,
            clf = clf,
            cv = cv
        )
        self.n_estimators = n_estimators
        self.clf = clf(
            n_estimators = self.n_estimators,
            random_state = self.random_state,
            class_weight = "balanced",
        )
        self.cv = cv(
            n_splits = self.folds,
            random_state = self.random_state,
            shuffle = True
        )
        self.sampling = sampling

    def fit(self, data: pd.DataFrame, Y: pd.DataFrame) -> None:
        sampled_X = data
        sampled_Y = Y

        if self.sampling is not None:
            sampling = self.sampling()
            sampled_X, sampled_Y = sampling.fit_resample(data, Y)
            sampled_X_train, sampled_X_test, sampled_Y_train, sampled_Y_test = train_test_split(
            sampled_X, 
            sampled_Y, 
            test_size = TEST_SET_SIZE, 
            stratify = sampled_Y
        )

        scores = cross_validate(
            estimator = self.clf,
            X = sampled_X,
            y = sampled_Y,
            cv = self.cv,
            scoring = self.scoring,
            n_jobs = self.n_jobs
        )

        X_train, X_test, Y_train, Y_test = train_test_split(
            data, 
            Y, 
            test_size = TEST_SET_SIZE,
            stratify = Y, 
            random_state = self.random_state
        )
        self.clf.fit(sampled_X_train, sampled_Y_train)
        self.results = {
            "cv": self._cv_name,
            "k_folds": self.folds,
            "cv_accuracy": scores["test_accuracy"].mean(),
            "cv_f1": scores["test_f1"].mean(),
            "cv_recall": scores["test_recall"].mean(),
            "cv_precision": scores["test_precision"].mean(),
            "train_score": self.clf.score(X_train, Y_train),
            "test_score": self.clf.score(X_test, Y_test),
        }
    
class DTClassifier(Classifier):
    def __init__(
        self, 
        max_features: str,
        random_state: int,  
        folds: int,
        scoring: Tuple[str], 
        n_jobs: int,
        clf: DecisionTreeClassifier, 
        cv: sklearn.model_selection.BaseCrossValidator,
    ) -> None:
        super().__init__(
            random_state = random_state,
            folds = folds,
            scoring = scoring,
            n_jobs = n_jobs,
            clf = clf,
            cv = cv
        )
        self.max_features = max_features
        self.clf = clf(
            max_features = self.max_features,
            random_state = self.random_state,
            class_weight = "balanced",
        )
        self.cv = cv(
            n_splits = self.folds,
            random_state = self.random_state,
            shuffle = True
        )

    def fit(self, data: pd.DataFrame, Y: pd.DataFrame) -> None:
        scores = cross_validate(
            estimator = self.clf,
            X = data,
            y = Y,
            cv = self.cv,
            scoring = self.scoring,
            n_jobs = self.n_jobs
        )

        X_train, X_test, Y_train, Y_test = train_test_split(
            data, 
            Y, 
            test_size = TEST_SET_SIZE,
            stratify = Y, 
            random_state = self.random_state
        )

        self.clf.fit(X_train, Y_train)

        self.results = {
            "cv": self._cv_name,
            "k_folds": self.folds,
            "cv_accuracy": scores["test_accuracy"].mean(),
            "cv_f1": scores["test_f1"].mean(),
            "cv_recall": scores["test_recall"].mean(),
            "cv_precision": scores["test_precision"].mean(),
            "train_score": self.clf.score(X_train, Y_train),
            "test_score": self.clf.score(X_test, Y_test),
        }
    
class Plot:
    def __init__(
        self, 
        clf: Classifier, 
        figsize: Tuple[int, int],
        save_name: str,  
        save_q: bool = True
    ) -> None:
        self.clf = clf
        self.save_name = save_name
        self.figsize = figsize
        self.save_q = save_q
        self.fig, self.ax = plt.subplots(1, figsize = self.figsize)

    def confusion_matrix(self, data: pd.DataFrame, Y: pd.DataFrame) -> None:
        cfm = sklearn.metrics.confusion_matrix(Y, self.clf.predict(data))
        sns.heatmap(
            cfm, 
            annot = True, 
            fmt = "d", 
            cmap = "Blues", 
            ax = self.ax,
            annot_kws = {"ha": "center", "va": "center"}
        )

    def decision_tree(
        tree: sklearn.tree.DecisionTreeClassifier,
        save_name: str,
        out_dir: str,
        feature_names: Iterable[str],
        class_names: Iterable[str] = ["no_error", "error"],
        figsize: tuple = (40, 30),
        save_q = True
    ) -> None:
        if not isinstance(self.clf, sklearn.tree.DecisionTreeClassifier):
            raise TypeError("Classifier is not a decision tree. Cannot plot it.")
        sklearn.tree.plot_tree(
            self.clf,
            feature_names = feature_names,
            class_names = class_names,
            fontsize = 14,
            ax = self.ax,
            filled = True
        )

    def _save(self, out_dir: str) -> None:
        if self.save_q:
            self.fig.savefig(
                os.path.join(out_dir, self.save_name),
                dpi = 100,
                bbox_inches = "tight"
            )
        plt.close(self.fig)

class IOHandler:
    def __init__(self, out_dir: str) -> None:
        self.out_dir = out_dir
        self.results_dir = os.path.join(self.out_dir, "qc_analysis_training_results")

    def _create_dir(self, name: str) -> str:
        dir_to_create = os.path.join(self.results_dir, name)
        if not os.path.exists(dir_to_create):
            os.makedirs(dir_to_create)
        return dir_to_create

    @staticmethod
    def _write(obj, out_dir: str) -> None:
        if hasattr(obj, "_save"):
            obj._save(out_dir)
        else:
            raise AttributeError(f"Object {obj} does not have a _save method.")
        
    def write_figs(self, plot: Plot) -> None:
        write_dir = self._create_dir("figs")
        self._write(plot, write_dir)

    def write_results(self, clf: Classifier) -> None:
        write_dir = self._create_dir("metrics")
        self._write(clf, write_dir)
        
def evaluate(trained_tree: sklearn.tree.DecisionTreeClassifier, data: pd.DataFrame):
    return trained_tree.predict(data)
