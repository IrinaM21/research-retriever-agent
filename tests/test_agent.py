import unittest
from agent import Agent

import logging

# Set up logging for traceability
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestAgent(unittest.TestCase):
    """Unit tests for the Agent class."""

    def setUp(self):
        self.agent = Agent(llm_name="microsoft/Phi-3.5-mini-instruct") # small model for testing

    def test_gen_from_prompt(self):

        # hallucination test
        prompt = "What is the capital of France?"
        response = self.agent.gen_from_prompt(prompt)
        self.assertIsInstance(response, str)
        self.assertIn("Paris", response)

    def test_route_query(self):
        logger.info("Testing route_query...")
        query = "What are three novel contributions to LLM research"
        self.agent.get_relevant_papers(query)  # Ensure relevant papers are set
        response = self.agent.route_query(query)
        self.assertIsInstance(response, str) # response should be a string
        self.assertGreater(len(response), 0) # should not be empty
        self.assertIn(response, ["creative", "technical"]) # should be one of the two categories
        self.assertNotIn(response, " ") # one word response only

    def test_answer_creative(self):
        query = "Suggest three future research directions in computer vision."
        self.agent.get_relevant_papers(query)  # Ensure relevant papers are set
        response = self.agent.answer_creative(query)
        print(repr(response)) # for debugging
        logger.info(f"Novelty detection response: {response}")
        self.assertIsInstance(response, str) # response should be a string
        self.assertGreater(len(response), 0) # should not be empty
        self.assertLess(len(response), 1000) # should be concise: just the 3-5 sentences requested in prompt

if __name__ == "__main__":
    unittest.main()