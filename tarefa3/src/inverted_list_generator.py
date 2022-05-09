import os
import logging
import sys
import xml.etree.ElementTree as ET
from nltk.tokenize import RegexpTokenizer
from unidecode import unidecode

class InvertedListGenerator:
    def __init__(self):
        self.configs = {}
        self.input_documents = {}
        self.inverted_list = {}
        
        logging.basicConfig(
            handlers=[
                logging.FileHandler("result/execution.log"),
                logging.StreamHandler(sys.stdout)
            ],
            format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s", 
            datefmt='%m/%d/%Y %H:%M:%S', 
            level=logging.DEBUG
        )

        logging.info("Starting Inverted List Generator.")

        self.read_config()

    def read_config(self):
        config_file = "gli.cfg"
        logging.info("Reading configurations' file.")

        with open(config_file, "r") as file:
            for line in file:
                line = line.strip().split("=")

                if (line[0] != "LEIA") and (line[0] != "ESCREVA"):
                    logging.warning("Unexpected parameter in configurations' file: " + line[0])
                    continue

                #print(line)

                if line[0] in self.configs:
                    self.configs[line[0]].append(line[1])
                else:
                    self.configs[line[0]] = [line[1]]

        #print(self.configs)

        if ("LEIA" not in self.configs) or ("ESCREVA" not in self.configs):
            logging.error("Malformed gli.cfg. 'LEIA' or 'ESCREVA' absent.")
            raise Exception("Malformed gli.cfg. 'LEIA' or 'ESCREVA' absent.")

        if len(self.configs["ESCREVA"]) > 1:
            logging.error("Malformed gli.cfg. 'ESCREVA' found more than once.")
            raise Exception("Malformed gli.cfg. 'ESCREVA' found more than once.")

    def read_input_files(self):
        num_documents = 0

        for path in self.configs["LEIA"]:
            logging.info("Reading file " + path)

            tree = ET.parse(path)
            root = tree.getroot()

            for document in root.iter("RECORD"):
                doc_identifier = document.find("RECORDNUM")
                doc_text = document.find("ABSTRACT")

                if doc_identifier is None:
                    logging.warning("Tag RECORDNUM not found in a document in file %s. Ignoring..." % (path))

                doc_identifier = int(doc_identifier.text.strip())

                if doc_text is None:
                    doc_text = document.find("EXTRACT")

                    if doc_text is None:
                        logging.warning("Document %d in file %s without ABSTRACT or EXTRACT. Ignoring..." % (doc_identifier, path))
                        continue

                doc_text = doc_text.text.strip()

                # logs reading document

                self.input_documents[doc_identifier] = doc_text

                num_documents += 1

            logging.info("Total documents read: " + str(num_documents))

    def generate_inverted_list(self):
        logging.info("Generating inverted list.")

        total_documents = 0

        for document_id in sorted(self.input_documents.keys()):
            text = self.input_documents[document_id].upper()
            tokenizer = RegexpTokenizer(r'[a-zA-Z]{3,}') #\w*
            words_list = tokenizer.tokenize(unidecode(text))

            for word in words_list:
                if word not in self.inverted_list:
                    self.inverted_list[word] = []

                self.inverted_list[word].append(document_id)

            total_documents += 1

        logging.info("Inverted list generated from %d documents, with %d terms considered." % (total_documents, len(self.inverted_list)))

    def save_inverted_list(self):
        logging.info("Saving inverted list to " + self.configs["ESCREVA"][0])

        total_terms = 0

        with open(self.configs["ESCREVA"][0], "w") as file:
            for term in sorted(self.inverted_list.keys()):
                indexes = ",".join(str(num) for num in sorted(self.inverted_list[term]))
                indexes = "[" + indexes + "]"

                file.write(term + ";" + indexes + "\n")

                total_terms += 1
        logging.info("%d terms written. Saved inverted list to '%s'" % (total_terms, self.configs["ESCREVA"][0]))

    def run(self):
        self.read_input_files()
        self.generate_inverted_list()
        self.save_inverted_list()

#inverted_list_generator = InvertedListGenerator()
#inverted_list_generator.run()