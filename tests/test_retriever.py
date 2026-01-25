import unittest
from retriever import Retriever

class TestRetriever(unittest.TestCase):
    """Unit tests for the Retriever class."""

    def setUp(self):
        self.retriever = Retriever()
    
    def test_semantic_search(self):
        query = "What are the latest advancements in natural language processing?"
        results = self.retriever.semantic_search(query)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)  # should return some results
        for res in results:
            self.assertIsInstance(res, int) # should return list of indices
    
    def test_keyword_search(self):
        query = "computer vision techniques"
        results = self.retriever.keyword_search(query)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)  # should return some results
        for res in results:
            self.assertIsInstance(res, int) # should return list of indices
    
    def test_hybrid_search(self):
        query = "deep learning applications in healthcare"
        results = self.retriever.hybrid_search(query)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)  # should return some results
        print(results[0])  # for debugging
        for res in results:
            self.assertIsInstance(res, str) # should return list of document texts