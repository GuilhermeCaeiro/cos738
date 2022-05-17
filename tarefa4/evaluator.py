import time
import sys
import os
import logging
import numpy as np
import pandas as pd
import math
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, average_precision_score
from sklearn.metrics import PrecisionRecallDisplay
import matplotlib.pyplot as plt

class Evaluator:

    def __init__(self, similarity_threshold=0.1):
        self.results_file_path = "result/resultados.csv"
        self.expected_results_file_path = "result/expected_results.csv"
        self.results = None
        self.expected_results = None
        self.similarity_threshold = similarity_threshold
        self.processed_results = {}
        self.merged_results = None

    def generate_unranked_result(merged_query_results):
        pass

    def generate_ranked_result():
        pass


    def read_results(self):
        self.results = pd.read_csv(self.results_file_path, sep=";", header=None, names=["query_id", "result_data"])
        self.expected_results = pd.read_csv(self.expected_results_file_path, sep=";")

        print(self.results.head())

        self.results[["rank", "document_id", "score"]] = self.results["result_data"].str.replace("[", "").str.replace("]", "").str.split(",", 2, expand=True)
        self.results["rank"] = pd.to_numeric(self.results["rank"])
        self.results["document_id"] = pd.to_numeric(self.results["document_id"])
        self.results["score"] = pd.to_numeric(self.results["score"])
        
        print(self.results.head())
        print(self.expected_results.head())
        print(self.results.dtypes)
        print(self.expected_results.dtypes)

        query_metrics = {}

        for query in sorted(self.results["query_id"].unique()):
            #print(query)

            query_results = None

            #if self.similarity_threshold > 0:
            #    query_results = self.results[(self.results["query_id"] == query) & (self.results["score"] > self.similarity_threshold)]
            #else:
            query_results = self.results[self.results["query_id"] == query]

            print(query_results)

            expected_query_result = self.expected_results[self.expected_results["QueryNumber"] == query]

            merged_query_results = pd.merge(query_results, expected_query_result, how="outer", left_on = "document_id", right_on = "DocNumber")

            #merged_query_results["result"] = (merged_query_results["score"] > 0.0) * 1
            #merged_query_results["expected_result"] = (merged_query_results["DocVotes"] > 0.0) * 1

            merged_query_results["result"] = merged_query_results["score"] > 0.0
            merged_query_results["expected_result"] = merged_query_results["DocVotes"] > 0.0

            print(merged_query_results)
            print(precision_score(merged_query_results["expected_result"], merged_query_results["result"]))
            print(recall_score(merged_query_results["expected_result"], merged_query_results["result"]))
            print(f1_score(merged_query_results["expected_result"], merged_query_results["result"]))

            if self.merged_results is None:
                self.merged_results = merged_query_results
            else:
                self.merged_results = pd.concat([self.merged_results, merged_query_results], ignore_index = True)

            print(merged_query_results.head(10))

            tmp = merged_query_results.loc[:9]
            print(tmp)

            query_metrics[query] = {
                "precision_at_5": self.precision_at_k(merged_query_results, 5),
                "precision_at_10": self.precision_at_k(merged_query_results, 10),
                "average_precision": self.average_precision(merged_query_results),
                "reciprocal_rank": self.reciprocal_rank(merged_query_results),
                "discounted_cumulative_gain": self.discounted_cumulative_gain(merged_query_results),
                
            }

            break


        print(query_metrics)


















    def eleven_point_precision_recall_curve(self):
        pass


    def f1_at_k(self, result, k=0):
        k = k if k > 0 else len(result)
        relevant_results = results.loc[:k-1]
        #print(relevant_results)
        return f1_score1(relevant_results["expected_result"], relevant_results["result"])

    def precision_at_k(self, results, k=5):
        relevant_results = results.loc[:k-1]
        #print(relevant_results)
        return precision_score(relevant_results["expected_result"], relevant_results["result"])

        

    def r_precision_histogram(self):
        pass

    def average_precision(self, results):
        precisions = []
        recalls = []
        indexes = []
        for index, row in results[(results["expected_result"] == True) & (results["result"] == True)].iterrows():
            tmp = results.loc[:index]
            precisions.append(precision_score(tmp["expected_result"], tmp["result"]))
            indexes.append(index)

        precisions = precisions + ([0.0] * len(results[(results["expected_result"] == True) & (results["result"] == False)]))
        #print(precisions)

        return np.mean(precisions)#, precisions, indexes

    def reciprocal_rank(self, results):
        relevant_docs_found = results[(results["expected_result"] == True) & (results["result"] == True)]

        if len(relevant_docs_found) == 0:
            return 0

        return 1/(relevant_docs_found.index.values.astype(int)[0] + 1)

    def discounted_cumulative_gain(self, results):
        pass

    def normalized_discounted_cumulative_gain(self):
        pass

    def generate_report(self):
        pass

    def run(self):
        self.read_results()
        #self.eleven_point_precision_recall_curve()

Evaluator().run()