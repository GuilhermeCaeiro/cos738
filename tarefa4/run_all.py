from src.inverted_list_generator import InvertedListGenerator
from src.indexer import Indexer
from src.searcher import Searcher
from src.query_processor import QueryProcessor
from src.evaluator import Evaluator

InvertedListGenerator().run()
Indexer().run()
QueryProcessor().run()
Searcher().run()
Evaluator().run()