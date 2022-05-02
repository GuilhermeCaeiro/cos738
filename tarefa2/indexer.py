import numpy as np
import logging

class Indexer:

    def __init__(self):
        self.configs = {}
        self.documents_matrix = None # matrix m x n, representing m documents and n words
        self.inverted_list = {} # dict with n keys (words)
        self.words_list = None
        self.document_occurences = {}
        self.number_of_documents = None
        self.document_ids = set()

        logging.basicConfig(filename='indexer.log', format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.DEBUG)

        self.read_config()

    def read_config(self):
        config_file = "index.cfg"
        logging.info("Reading configurations' file.")

        with open(config_file, "r") as file:
            for line in file:
                line = line.strip().split("=")

                if (line[0] != "LEIA") and (line[0] != "ESCREVA"):
                    logging.warning("Unexpected parameter in configurations' file: " + line[0])
                    continue

                print(line)

                if line[0] in self.configs:
                    self.configs[line[0]].append(line[1])
                else:
                    self.configs[line[0]] = [line[1]]

        print(self.configs)

        if ("LEIA" not in self.configs) or ("ESCREVA" not in self.configs):
            logging.error("Malformed gli.cfg. 'LEIA' or 'ESCREVA' absent.")
            raise Exception("Malformed gli.cfg. 'LEIA' or 'ESCREVA' absent.")

        if (len(self.configs["LEIA"]) > 1) or (len(self.configs["ESCREVA"]) > 1):
            logging.error("Malformed gli.cfg. 'LEIA' or 'ESCREVA' found more than once.")
            raise Exception("Malformed gli.cfg. 'LEIA' or 'ESCREVA' found more than once.")

    def load_inverted_list(self):
        logging.info("Loading inverted list.")
        with open(self.configs["LEIA"][0], "r") as file:
            for line in file:
                splited_line = line.strip().split(";")
                word = splited_line[0]
                document_ids = eval(splited_line[1])
                self.inverted_list[word] = document_ids

        logging.info("Inverted list loaded.")

    def retrieve_words_list(self):
        self.words_list = sorted(self.inverted_list.keys())

    def calculate_tf(self):
        logging.info("Calculating tf (term frequency) for all documents.")
        tf_matrix = np.zeros((self.number_of_documents, len(self.words_list)), dtype=float)

        for column in range(0, len(self.words_list)):
            word = self.words_list[column]
            for document_id in self.inverted_list[word]:
                line = self.document_ids.index(document_id) # document_ids should have no repetitions, because it was a set.
                tf_matrix[line,column] += 1

        for i in range(self.number_of_documents):
            tf_matrix[i, :] = tf_matrix[i, :] / max(tf_matrix[i, :])

        logging.info("Finished calculating tf for all documents.")

        return tf_matrix

    def calculate_idf(self):
        logging.info("Calculating idf (inverse document frequency) for all documents.")

        idfs = []

        for word in self.words_list:
            documents = set(self.inverted_list[word])
            self.document_ids.update(documents) # adds document ids to the set of known documents

            idfs.append(len(documents)) # adds the number of documents with a given word

        self.document_ids = sorted(list(self.document_ids))

        self.number_of_documents = len(self.document_ids)

        idfs = np.log(self.number_of_documents / np.array([idfs])) # the log will be calculated for each member of the matrix

        logging.info("Finished calculating idf for all documents.")

        return idfs

            

    def calculate_tf_idf(self):
        logging.info("Calculating matrix of the TF-IDFs.")

        idf = self.calculate_idf()
        tf = self.calculate_tf()

        print(tf.shape, idf.shape, np.diag(idf).shape)

        self.documents_matrix = tf * idf # (m x n) * (1 x n). As they are ndarrays, each line in tf will be multiplied by idf row wise, keeping a m x n ndarray as result.

        print(self.documents_matrix.shape)

        logging.info("Finished calculating matrix of the TF-IDFs.")

    def save_model(self):
        logging.info("Saving the TF-IDF matrix.")
        with open(self.configs["ESCREVA"][0], "wb") as file:
            np.save(file, self.documents_matrix)

        logging.info("TF-IDFs matrix saved to file: " + self.configs["ESCREVA"][0])

    def run(self):
        self.load_inverted_list()
        self.retrieve_words_list()
        self.calculate_tf_idf()
        self.save_model()

indexer = Indexer()
indexer.run()
