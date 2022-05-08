import numpy as np
import logging
import pickle
import time
from nltk.tokenize import RegexpTokenizer
from unidecode import unidecode

class Indexer:

    def __init__(self):
        self.configs = {}
        self.documents_matrix = None # matrix m x n, representing m documents and n words
        self.inverted_list = {} # dict with n keys (words)
        self.words_list = None
        self.document_occurences = {}
        self.number_of_documents = None
        self.document_ids = set()

        logging.basicConfig(filename='execution.log', format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s", datefmt='%m/%d/%Y %H:%M:%S', level=logging.DEBUG)

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
        start_time = time.time()

        tf_matrix = np.zeros((self.number_of_documents, len(self.words_list)), dtype=float)

        for column in range(0, len(self.words_list)):
            word = self.words_list[column]
            for document_id in self.inverted_list[word]:
                line = self.document_ids.index(document_id) # document_ids should have no repetitions, because it was a set.
                tf_matrix[line,column] += 1

        for i in range(self.number_of_documents):
            tf_matrix[i, :] = tf_matrix[i, :] / max(tf_matrix[i, :])

        logging.info("Finished calculating tf for all documents. Total time: %f seconds." % (time.time() - start_time))

        return tf_matrix

    def calculate_idf(self):
        logging.info("Calculating idf (inverse document frequency) for all documents.")

        idfs = []
        start_time = time.time()

        for word in self.words_list:
            documents = set(self.inverted_list[word])
            self.document_ids.update(documents) # adds document ids to the set of known documents

            idfs.append(len(documents)) # adds the number of documents with a given word

        self.document_ids = sorted(list(self.document_ids))

        self.number_of_documents = len(self.document_ids)

        idfs = np.log(self.number_of_documents / np.array([idfs])) # the log will be calculated for each member of the matrix

        logging.info("Finished calculating idf for all documents. Total time: %f seconds." % (time.time() - start_time))

        return idfs

            

    def calculate_tf_idf(self):
        logging.info("Calculating matrix of the TF-IDFs.")

        start_time = time.time()

        self.idf = self.calculate_idf()
        self.tf = self.calculate_tf()

        print(self.tf.shape, self.idf.shape, np.diag(self.idf).shape)

        self.documents_matrix = self.tf * self.idf # (m x n) * (1 x n). As they are ndarrays, each line in tf will be multiplied by idf row wise, keeping a m x n ndarray as result.

        print(self.documents_matrix.shape)

        logging.info("Finished calculating matrix of the TF-IDFs. Total time: %f seconds." % (time.time() - start_time))

    def save_model(self):
        logging.info("Saving the TF-IDF matrix.")

        model = VectorModel(
            self.words_list, 
            self.document_ids, 
            self.documents_matrix, 
            self.tf, 
            self.idf
        )
        
        with open(self.configs["ESCREVA"][0], "wb") as file:
            #np.save(file, self.documents_matrix)
            pickle.dump(model, file)

        logging.info("TF-IDFs matrix saved to file: " + self.configs["ESCREVA"][0])

    def run(self):
        self.load_inverted_list()
        self.retrieve_words_list()
        self.calculate_tf_idf()
        self.save_model()


class VectorModel:
    def __init__(self, words_list, document_ids, documents_matrix, tf, idf, threshold = 0.7071, use_thresold = False):
        self.words_list = words_list
        self.documents_matrix = documents_matrix
        self.document_ids = document_ids
        self.use_thresold = use_thresold
        self.threshold = threshold
        self.tf = tf
        self.idf = idf

    def generate_query_vector(self, query):
        text = query.upper()
        tokenizer = RegexpTokenizer(r'[a-zA-Z]{3,}') #\w*
        query_words_list = tokenizer.tokenize(unidecode(text))

        query_vector = np.zeros(len(self.words_list))

        query_words_frequency = {}

        for word in query_words_list:
            if word not in query_words_frequency:
                query_words_frequency[word] = 1
            else:
                query_words_frequency[word] += 1

        max_word_frequency = max(query_words_frequency.values())
        sum_words_frequency = sum(query_words_frequency.values())

        for word in query_words_frequency.keys():
            word_position = None

            try:
                word_position = self.words_list.index(word)
            except ValueError:
                print("Word %s present in the query but not present in the words list." % (word))
                continue

            weight = query_words_frequency[word] / sum_words_frequency
            #weight = (0.5 * ((0.5 * query_words_frequency[word]) / max_word_frequency)) * self.idf[word_position]
            #print(len(query_words_frequency), self.idf.shape)

            query_vector[word_position] = weight

        return query_vector


    def generate_result(self, similarity, threshold):
        if threshold is None:
            threshold = self.threshold

        partial_results = {}
        results = []
        rank_position = 0

        print(type(similarity), len(similarity))

        for i in range(len(similarity)):
            if self.use_thresold and similarity[i] < threshold:
                continue

            partial_results[self.document_ids[i]] = similarity[i]

        for document_id in sorted(partial_results, key = partial_results.get, reverse = True):
            rank_position += 1
            results.append([rank_position, document_id, partial_results[document_id]])

        return results

    def calculate_similarity(self, query_vector):
        similarity = []

        for i in range(len(self.document_ids)):
            similarity.append(sum(self.documents_matrix[i, :] * query_vector)/(np.linalg.norm(self.documents_matrix[i, :]) * np.linalg.norm(query_vector)))

        return similarity



    def evaluate_query(self, query, threshold = None):
        query_vector = self.generate_query_vector(query)
        similarity = self.calculate_similarity(query_vector)
        return self.generate_result(similarity, threshold)







#indexer = Indexer()
#indexer.run()
