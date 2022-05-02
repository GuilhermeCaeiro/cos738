import numpy as np

class indexer:

    def __init__(self):
        self.documents_matrix = None # matrix m x n, representing m documents and n words
        self.inverted_list = None # dict with n keys (words)
        self.words_list = None
        self.document_occurences = {}
        self.number_of_documents = None
        self.document_ids = set()

    def retrieve_words_list(self):
        self.words_list = sorted(self.inverted_list.keys())

    def load_inverted_list(self):
        pass

    def calculate_tf():
        tf_matrix = np.zeros((self.number_of_documents, len(self.words_list)), dtype=float)

        for column in range(0, len(self.words_list)):
            word = self.words_list[column]
            for document_id in self.inverted_list[word]:
                line = self.document_ids.index(document_id) # document_ids should have no repetitions, because it was a set.
                tf_matrix[line,column] += 1

        for i in range(self.number_of_documents):
            tf_matrix[i, :] = tf_matrix[i, :] / max(tf_matrix[i, :])

        return tf_matrix

    def calculate_idf():
        idfs = []

        for word in words_list:
            documents = set(words_list[word])
            self.document_ids.update(documents) # adds document ids to the set of known documents

            idfs.append(len(documents)) # adds the number of documents with a given word

        self.document_ids = sorted(list(self.document_ids))

        self.number_of_documents = len(self.document_ids)

        idfs = np.log(self.number_of_documents / np.array([idfs])) # the log will be calculated for each member of the matrix

        return idfs

            

    def calculate_tf_idf(self):
        tf = self.calculate_tf()
        idf = self.calculate_idf()

        self.documents_matrix = np.matrix(tf) * np.matrix(np.diag(idf)) # conveting matrices to matrix to avoid the weird behavior of ndarray.
