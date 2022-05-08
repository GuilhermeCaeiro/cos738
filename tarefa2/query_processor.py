import os
import logging
import xml.etree.ElementTree as ET
from nltk.tokenize import RegexpTokenizer
from unidecode import unidecode

class QueryProcessor:
    def __init__(self):
        self.configs = {}
        self.queries = {}

        logging.basicConfig(filename='execution.log', format='[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.DEBUG)

        self.read_config()

    def read_config(self):
        config_file = "pc.cfg"
        logging.info("Reading configurations' file '%s'." % config_file)

        with open(config_file, "r") as file:
            for line in file:
                line = line.strip().split("=")

                if (line[0] != "LEIA") and (line[0] != "CONSULTAS") and (line[0] != "ESPERADOS"):
                    logging.warning("Unexpected parameter in configurations' file: " + line[0])
                    continue

                print(line)

                if line[0] in self.configs:
                    self.configs[line[0]].append(line[1])
                else:
                    self.configs[line[0]] = [line[1]]

        print(self.configs)

        if ("LEIA" not in self.configs) or ("CONSULTAS" not in self.configs) or ("ESPERADOS" not in self.configs):
            logging.error("Malformed gli.cfg. 'LEIA' or 'CONSULTAS' or 'ESPERADOS' absent.")
            raise Exception("Malformed gli.cfg. 'LEIA' or 'CONSULTAS' or 'ESPERADOS' absent.")

        if (len(self.configs["LEIA"]) > 1) or (len(self.configs["CONSULTAS"]) > 1) or (len(self.configs["ESPERADOS"]) > 1):
            logging.error("Malformed gli.cfg. 'LEIA' or 'CONSULTAS' or 'ESPERADOS' found more than once.")
            raise Exception("Malformed gli.cfg. 'LEIA' or 'CONSULTAS' or 'ESPERADOS' found more than once.")

    def process_query_text(self, query):
        return unidecode(query.upper())

    def read_and_write_queries(self):
        query_file = self.configs["LEIA"][0]
        logging.info("Reading file " + query_file)

        processed_queries_file = open(self.configs["CONSULTAS"][0], "w")
        processed_queries_file.write("QueryNumber;QueryText\n")

        tree = ET.parse(query_file)
        root = tree.getroot()

        total_queries = 0

        for query in root.iter("QUERY"):
            query_number = int(query.find("QueryNumber").text)
            query_text = query.find("QueryText").text.strip().replace("\n", "").replace(";", "")
            num_results = int(query.find("Results").text)
            
            query_text = self.process_query_text(query_text)

            self.queries[query_number] = {
                "query_text": query_text,
                "num_results": num_results
            }

            processed_queries_file.write(str(query_number) + ";" + query_text + "\n")

            expected_results = {}

            for item in query.iter("Item"):
                expected_result = int(item.text)
                expected_score = 0

                for character in item.get("score"):
                    #if int(character) > 0:
                    #    expected_score += 1
                    expected_score += int(character)

                expected_results[expected_result] = expected_score

            self.queries[query_number]["expected_results"] = expected_results

            total_queries += 1


        processed_queries_file.close()

        logging.info("%d queries read and written to %s." % (total_queries, self.configs["CONSULTAS"][0]))

    def write_expected_results(self):
        logging.info("Writing expected results.")
        if len(self.queries.keys()) == 0:
            print("Method 'read_and_write_queries' must be executed first.")
            logging.error("Method 'read_and_write_queries' must be executed first.")
            return

        expected_results_file = open(self.configs["ESPERADOS"][0], "w")
        expected_results_file.write("QueryNumber;DocNumber;DocVotes\n")

        total_results = 0

        for query_number in sorted(self.queries.keys()):
            expected_results = self.queries[query_number]["expected_results"]
            for document_number in sorted(expected_results, key = expected_results.get, reverse = True):
                expected_results_file.write(str(query_number) + ";" + str(document_number) + ";" + str(expected_results[document_number]) + "\n")

                total_results += 1

        expected_results_file.close()

        logging.info("%d expected results written to '%s'." % (total_results, self.configs["ESPERADOS"][0]))
    
    def run(self):
        self.read_and_write_queries()
        self.write_expected_results()

#query_processor = QueryProcessor()
#query_processor.run()