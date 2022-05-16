import time
import sys
import os
import logging
import numpy as np
import pandas as pd
import math

class Evaluator:

    def __init__(self):
        self.results_file_path = "result/resultados.csv"
        self.expected_results_file_path = "result/expected_results.csv"
        self.results = None
        self.expected_results = None


    def read_results(self):
        self.results = pd.read_csv(self.results_file_path, sep=";", header=None, names=["query_id", "result_data"])
        self.expected_results = pd.read_csv(self.expected_results_file_path, sep=";")

        print(self.results.head())

        self.results[["rank", "document_id", "score"]] = self.results["result_data"].str.replace("[", "").str.replace("]", "").str.split(",", 2, expand=True)
        print(self.results.head())
        print(self.expected_results.head())

    def eleven_point_precision_recall_curve(self):
        pass

    def f1(self):
        pass

    def precision_at_5(self):
        pass

    def precision_at_10(self):
        pass

    def r_precision_histogram(self):
        pass

    def mean_average_precision(self):
        pass

    def mean_reciprocal_rank(self):
        pass

    def discounted_cumulative_gain(self):
        pass

    def normalized_discounted_cumulative_gain(self):
        pass

    def generate_report(self):
        pass

    def run(self):
        self.read_results()

Evaluator().run()