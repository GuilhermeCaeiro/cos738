import time
import sys
import os
import logging
import numpy as np
import pandas as pd
import math
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, average_precision_score, dcg_score, ndcg_score
from sklearn.metrics import PrecisionRecallDisplay
#import matplotlib.pyplot as plt

class Evaluator:

    def __init__(self, similarity_threshold=0.1):
        self.results_file_path = "result/resultados.csv"
        self.expected_results_file_path = "result/expected_results.csv"
        self.results = None
        self.expected_results = None
        self.similarity_threshold = similarity_threshold
        self.processed_results = {}
        #self.merged_results = None
        self.using_stemmer = False
        self.query_metrics = {}
        self.metrics_summary = {}
        self.output_dir = "avalia/"
        self.metrics = [
            "eleven_point_precision_recall_curve",
            #"f1_at_10",
            "f1",
            "precision_at_5",
            "precision_at_10",
            "r_precision_histogram",
            "average_precision",
            "reciprocal_rank",
            "discounted_cumulative_gain",
            "normalized_discounted_cumulative_gain",
        ]

        logging.basicConfig(
            handlers=[
                logging.FileHandler("result/execution.log"),
                logging.StreamHandler(sys.stdout)
            ],
            format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s", 
            datefmt='%m/%d/%Y %H:%M:%S', 
            level=logging.DEBUG
        )

        self.check_stemmer()

    def check_stemmer(self):
        # uses gli.cfg as reference.
        with open("gli.cfg") as file:
            for line in file:
                #print(line)
                if line.strip().lower() == "stemmer":
                    logging.info("Uso de stemmer detectado.")
                    self.using_stemmer = True


    def read_results(self):
        logging.info("Evaluating results.")
        #print(self.using_stemmer)
        self.results = pd.read_csv(self.results_file_path, sep=";", header=None, names=["query_id", "result_data"])
        self.expected_results = pd.read_csv(self.expected_results_file_path, sep=";")

        #print(self.results.head())

        self.results[["rank", "document_id", "score"]] = self.results["result_data"].str.replace("[", "").str.replace("]", "").str.split(",", 2, expand=True)
        self.results["rank"] = pd.to_numeric(self.results["rank"])
        self.results["document_id"] = pd.to_numeric(self.results["document_id"])
        self.results["score"] = pd.to_numeric(self.results["score"])
        
        #print(self.results.head())
        #print(self.expected_results.head())
        #print(self.results.dtypes)
        #print(self.expected_results.dtypes)

        query_metrics = {}

        for query in sorted(self.results["query_id"].unique()):
            logging.info("Evaluating query %d." % query)
            #print(query)

            query_results = None

            #if self.similarity_threshold > 0:
            #    query_results = self.results[(self.results["query_id"] == query) & (self.results["score"] > self.similarity_threshold)]
            #else:
            query_results = self.results[self.results["query_id"] == query]

            #print(query_results)

            expected_query_result = self.expected_results[self.expected_results["QueryNumber"] == query]

            merged_query_results = pd.merge(query_results, expected_query_result, how="outer", left_on = "document_id", right_on = "DocNumber")

            #merged_query_results["result"] = (merged_query_results["score"] > 0.0) * 1
            #merged_query_results["expected_result"] = (merged_query_results["DocVotes"] > 0.0) * 1

            merged_query_results["result"] = merged_query_results["score"] > 0.0
            merged_query_results["expected_result"] = merged_query_results["DocVotes"] > 0.0
            merged_query_results[["score", "DocVotes"]] = merged_query_results[["score", "DocVotes"]].fillna(value=0)

            #print(merged_query_results)
            #print(precision_score(merged_query_results["expected_result"], merged_query_results["result"]))
            #print(recall_score(merged_query_results["expected_result"], merged_query_results["result"]))
            #print(f1_score(merged_query_results["expected_result"], merged_query_results["result"]))

            #if self.merged_results is None:
            #    self.merged_results = merged_query_results
            #else:
            #    self.merged_results = pd.concat([self.merged_results, merged_query_results], ignore_index = True)

            #print(merged_query_results.head(10))

            tmp = merged_query_results.loc[:9]
            #print(tmp)

            query_metrics[query] = {
                "f1_at_10": self.f1_at_k(merged_query_results, 10),
                "f1": self.f1_at_k(merged_query_results),
                "precision_at_5": self.precision_at_k(merged_query_results, 5),
                "precision_at_10": self.precision_at_k(merged_query_results, 10),
                "average_precision": self.average_precision(merged_query_results),
                "reciprocal_rank": self.reciprocal_rank(merged_query_results),
                "discounted_cumulative_gain": self.discounted_cumulative_gain(merged_query_results, 10),
                "normalized_discounted_cumulative_gain": self.normalized_discounted_cumulative_gain(merged_query_results, 10),
            }

            #break


        #print(query_metrics)
        self.query_metrics = query_metrics

        logging.info("Result evaluation finished.")


    def eleven_point_precision_recall_curve(self):
        pass


    def f1_at_k(self, results, k=0):
        k = k if k > 0 else len(results)
        relevant_results = results.loc[:k-1]
        #print(relevant_results)
        return f1_score(relevant_results["expected_result"], relevant_results["result"])

    def precision_at_k(self, results, k):
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

    def discounted_cumulative_gain(self, results, k=10):
        ground_truth = np.array([list(results["DocVotes"])])
        estimated = np.array([list(results["score"])])
        return dcg_score(ground_truth, estimated, k=k)


    def normalized_discounted_cumulative_gain(self, results, k=10):
        ground_truth = np.array([list(results["DocVotes"])])
        estimated = np.array([list(results["score"])])
        return ndcg_score(ground_truth, estimated, k=k)

    def generate_files(self):
        logging.info("Generating CSVs.")
        for metric in self.metrics:
            stemmer = "stemmer" if self.using_stemmer else "nostemmer"
            outputfile = open(self.output_dir + metric + "-" + stemmer + "-1.csv", "w")
            outputfile.write("query_id," + metric + "\n")

            metric_values = []

            for query_id in sorted(self.query_metrics.keys()):
                if metric not in self.query_metrics[query_id]:
                    continue
                #print(metric, self.query_metrics[query_id], self.query_metrics[query_id][metric])
                metric_values.append(self.query_metrics[query_id][metric])
                line = str(query_id) + "," + str(self.query_metrics[query_id][metric]) + "\n"
                outputfile.write(line)

            self.metrics_summary[metric] = metric_values

            outputfile.close()
            logging.info("CSVs generated.")

    def generate_report(self):
        logging.info("Generating report file.")
        with open("RELATORIO.MD", "a") as file:
            file.write("# Avaliação dos Resultados\n\n")
            file.write("### Avaliação dos Resultados do modelo vetorial sob múltiplas métricas.\n\n")
            file.write("Total de queries: %f\n\n" % len(self.query_metrics))
            file.write("Utilizando Porter Stemmer? %s\n\n" % ("Sim" if self.using_stemmer else "Não"))
            file.write("## 11-point precision recall curve\n\n")
            file.write("Não implementado.\n\n")
            file.write("## F1 Score\n\n")
            file.write("Média dos F1 Scores computados sobre todas a queries: %f\n\n" % (np.mean(self.metrics_summary["f1"])))
            file.write("## Precision@5\n\n")
            file.write("Média das Precision@5 computadas sobre todas a queries: %f\n\n" % (np.mean(self.metrics_summary["precision_at_5"])))
            file.write("## Precision@10\n\n")
            file.write("Média das Precision@10 computads sobre todas a queries: %f\n\n" % (np.mean(self.metrics_summary["precision_at_10"])))
            file.write("## Histograma de R-Precision\n\n")
            file.write("Não implementado.\n\n")
            file.write("## Mean Average Precision\n\n")
            file.write("Mean Average Precision computada sobre todas a queries: %f\n\n" % (np.mean(self.metrics_summary["average_precision"])))
            file.write("## Mean Reciprocal Rank\n\n")
            file.write("Mean Reciprocal Rank computado sobre todas a queries: %f\n\n" % (np.mean(self.metrics_summary["reciprocal_rank"])))
            file.write("## Discounted Cumulative Gain\n\n")
            file.write("Média do Discounted Cumulative Gain computado sobre todas a queries: %f\n\n" % (np.mean(self.metrics_summary["discounted_cumulative_gain"])))
            file.write("## Normalized Discounted Cumulative Gain\n\n")
            file.write("Média do Normalized Discounted Cumulative Gain computado sobre todas a queries: %f\n\n" % (np.mean(self.metrics_summary["normalized_discounted_cumulative_gain"])))
            file.write("#### Métricas por query podem ser verificadas nos arquivos CSV presentes no diretório \"avalia\"\n\n")
            file.write("---\n\n")

            logging.info("Report file generated.")

    def run(self):
        self.read_results()
        self.generate_files()
        self.generate_report()

#Evaluator().run()