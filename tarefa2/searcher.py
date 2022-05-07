import numpy as np
import logging
import pickle

class Searcher:

    def __init__(self):
        self.configs = {}
        self.model = None
        self.queries = {}

        logging.basicConfig(filename='indexer.log', format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.DEBUG)

        self.read_config()

    def read_config(self):
        config_file = "index.cfg"
        logging.info("Reading configurations' file.")

        with open(config_file, "r") as file:
            for line in file:
                line = line.strip().split("=")

                if (line[0] != "MODELO") and (line[0] != "CONSULTAS") and (line[0] != "RESULTADOS"):
                    logging.warning("Unexpected parameter in configurations' file: " + line[0])
                    continue

                print(line)

                if line[0] in self.configs:
                    self.configs[line[0]].append(line[1])
                else:
                    self.configs[line[0]] = [line[1]]

        print(self.configs)

        if ("MODELO" not in self.configs) or ("CONSULTAS" not in self.configs) or ("RESULTADOS" not in self.configs):
            logging.error("Malformed gli.cfg. 'MODELO' or 'CONSULTAS' or 'RESULTADOS' absent.")
            raise Exception("Malformed gli.cfg. 'MODELO' or 'CONSULTAS' or 'RESULTADOS' absent.")

        if (len(self.configs["MODELO"]) > 1) or (len(self.configs["CONSULTAS"]) > 1) or (len(self.configs["RESULTADOS"]) > 1):
            logging.error("Malformed gli.cfg. 'MODELO' or 'CONSULTAS' or 'RESULTADOS' found more than once.")
            raise Exception("Malformed gli.cfg. 'MODELO' or 'CONSULTAS' or 'RESULTADOS' found more than once.")

    def load_model(self):
        with open(self.configs["MODELO"][0],'rb') as file: 
            self.model = pickle.load(file)

    def load_queries(self):
        with open(self.configs["CONSULTAS"][0],'r') as file:
            for line in  file:
                self.queries.append() 