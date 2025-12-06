import unittest
from retriever import Retrieve

class TestRetriever(unittest.TestCase):
    """Unit tests for the Retriever class."""

    def setUp(self):
        self.retriever = Retrieve()