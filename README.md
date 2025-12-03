# research-retriever-agent
Hybrid retriever and reasoning agent for analyzing scientific abstracts

## Components:

1. retriever.py: Retrieval system using BM25 keyword search and semantic search, combined with Reciprocal Rank Fusion
2. agent.py: Router model that filters queries into two subtasks, Small LLM (Phi-3.5-mini-instruct, loaded with Hugging Face) generates answers

### BM25

Ranks documents according to keyword frequency. Keywords that appear frequently contribute less to document rank, and scores are 
normalized for document length. Both keyword frequency and document normalization are adjustable.

### Semantic search

Documents are embedded, and ranked by proximity (cosine similarity or inner product) in vector space. In this implementation, embeddings are 
generated with Hugging Face, and the vector database with FAISS.

### Reciprocal rank fusion

Combines multiple ranked lists into one by summing the reciprocals of each document's ranks (rewards high rankings on each list). In this 
implementation, a constant K is added in the denominator to prevent top-ranked results from dominating the ranking system.

### Pipeline

Retrieve relevant papers -> determine subtask -> execute subtask

## Installation:
`pip install -r requirements.txt`

## Usage:
```
from agent import Agent

agent = Agent()

query = "Can you summarize what's novel about this abstract?"

response = agent.run(query)

print(response)

```
