import unittest
from agent import Agent

class TestAgent(unittest.TestCase):
    """Unit tests for the Agent class."""

    def setUp(self):
        self.agent = Agent()

    def test_gen_from_prompt(self):

        # hallucination test
        prompt = "What is the capital of France?"
        response = agent.gen_from_prompt(prompt)
        self.assertIsInstance(response, str)
        self.assertIn("Paris", response)

    def test_route_query(self):
        query = """Can you summarize the novel ideas in this abstract? 
        We introduce Mixtral 8x7B, a Sparse Mixture of Experts (SMoE) language model. 
        Mixtral has the same architecture as Mistral 7B, with the difference that each layer 
        is composed of 8 feedforward blocks (i.e. experts). For every token, at each layer, a 
        router network selects two experts to process the current state and combine their outputs. 
        Even though each token only sees two experts, the selected experts can be different at each timestep. 
        As a result, each token has access to 47B parameters, but only uses 13B active parameters during inference. 
        Mixtral was trained with a context size of 32k tokens and it outperforms or matches Llama 2 70B and GPT-3.5 
        across all evaluated benchmarks. In particular, Mixtral vastly outperforms Llama 2 70B on mathematics, code 
        generation, and multilingual benchmarks. We also provide a model fine-tuned to follow instructions, 
        Mixtral 8x7B - Instruct, that surpasses GPT-3.5 Turbo,"""
        response = self.agent.route_query(query)
        self.assertIsInstance(response, str) # response should be a string
        self.assertGreater(len(response), 0) # should not be empty
        self.assertIn(response, ["novelty", "direction"]) # should be one of the two categories
        self.assertLess(len(response), 9) # one word response only

    def test_find_novelty(self):
        query = """Can you summarize the novel ideas in this abstract? 
        We introduce Mixtral 8x7B, a Sparse Mixture of Experts (SMoE) language model. 
        Mixtral has the same architecture as Mistral 7B, with the difference that each layer 
        is composed of 8 feedforward blocks (i.e. experts). For every token, at each layer, a 
        router network selects two experts to process the current state and combine their outputs. 
        Even though each token only sees two experts, the selected experts can be different at each timestep. 
        As a result, each token has access to 47B parameters, but only uses 13B active parameters during inference. 
        Mixtral was trained with a context size of 32k tokens and it outperforms or matches Llama 2 70B and GPT-3.5 
        across all evaluated benchmarks. In particular, Mixtral vastly outperforms Llama 2 70B on mathematics, code 
        generation, and multilingual benchmarks. We also provide a model fine-tuned to follow instructions, 
        Mixtral 8x7B - Instruct, that surpasses GPT-3.5 Turbo,"""
        response = self.agent.find_novelty(query)
        self.assertIsInstance(response, str) # response should be a string
        self.assertGreater(len(response), 0) # should not be empty
        self.assertLess(len(response), 1000) # should be concise: just the 3-5 sentences requested in prompt