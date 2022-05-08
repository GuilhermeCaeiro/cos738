import numpy as np
import logging
import pickle
import time
from indexer import VectorModel

class Searcher:

    def __init__(self):
        self.configs = {}
        self.model = None
        self.queries = {}

        logging.basicConfig(filename='result/execution.log', format='[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.DEBUG)

        self.read_config()

    def read_config(self):
        config_file = "busca.cfg"
        logging.info("Reading configurations' file '%s'." % config_file)

        with open(config_file, "r") as file:
            for line in file:
                line = line.strip().split("=")

                if (line[0] != "MODELO") and (line[0] != "CONSULTAS") and (line[0] != "RESULTADOS"):
                    logging.warning("Unexpected parameter in configurations' file: " + line[0])
                    continue

                #print(line)

                if line[0] in self.configs:
                    self.configs[line[0]].append(line[1])
                else:
                    self.configs[line[0]] = [line[1]]

        #print(self.configs)

        if ("MODELO" not in self.configs) or ("CONSULTAS" not in self.configs) or ("RESULTADOS" not in self.configs):
            logging.error("Malformed gli.cfg. 'MODELO' or 'CONSULTAS' or 'RESULTADOS' absent.")
            raise Exception("Malformed gli.cfg. 'MODELO' or 'CONSULTAS' or 'RESULTADOS' absent.")

        if (len(self.configs["MODELO"]) > 1) or (len(self.configs["CONSULTAS"]) > 1) or (len(self.configs["RESULTADOS"]) > 1):
            logging.error("Malformed gli.cfg. 'MODELO' or 'CONSULTAS' or 'RESULTADOS' found more than once.")
            raise Exception("Malformed gli.cfg. 'MODELO' or 'CONSULTAS' or 'RESULTADOS' found more than once.")

    def load_model(self):
        logging.info("Loading model.")
        with open(self.configs["MODELO"][0],'rb') as file: 
            self.model = pickle.load(file)
        logging.info("Model loaded.")

    def load_queries(self):
        logging.info("Loading queries.")
        total_queries = 0
        with open(self.configs["CONSULTAS"][0],'r') as file:
            header_line = True
            for line in file:
                if header_line:
                    header_line = False
                    continue
                query_data = line.split(";")
                self.queries[query_data[0]] = query_data[1]

                total_queries += 1
        logging.info("%d queries loaded." % total_queries)

    def run_queries(self):
        logging.info("Running queries.")
        total_queries = 0
        start_time = time.time()
        individual_query_times = []

        with open(self.configs["RESULTADOS"][0],'w') as file:
            for query_id in sorted(self.queries.keys()):
                query_start_time = time.time()

                # query_results -> [rank_position, document_id, partial_results[document_id]]
                query_results = self.model.evaluate_query(self.queries[query_id])

                individual_query_times.append(time.time() - query_start_time)

                for result in query_results:
                    file.write(str(query_id) + ";[" + str(result[0]) + "," + str(result[1]) + "," + str(result[2]) + "]\n")

                total_queries += 1

        logging.info("%d queries executed in %f seconds, with average query execution time of %f seconds." % (total_queries, time.time() - start_time, np.mean(individual_query_times)))

    def run(self):
        self.load_model()
        self.load_queries()
        self.run_queries()


#searcher = Searcher()
#searcher.run()

