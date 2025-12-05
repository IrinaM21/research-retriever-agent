# research-retriever-agent
Hybrid retriever and reasoning agent for analyzing scientific abstracts

## Components:

1. retriever.py: Retrieval system using BM25 keyword search and semantic search, combined with Reciprocal Rank Fusion
2. agent.py: Router model that filters queries into two subtasks, Small LLM (Phi-3.5-mini-instruct, loaded with Hugging Face) generates answers

### BM25

Ranks documents according to keyword frequency. Keywords that appear frequently contribute less to document rank, and scores are 
normalized for document length. Both keyword saturation and length normalization are adjustable.

This repo uses the BM25 implementation from the `bm25s` library. While it requires more memory than other implementations,  `bm25s` 
enables much faster retrieval:
In `bm25s`, term frequency, inverse document frequencies, and length normalization (calculated using average document length + the 
tunable length normalization and keyword saturation parameters) are computed and combined into a score for each document-keyword 
pair. These scores are stored in a Scipy sparse matrix prior to query time. Rather than calculating the overall document score at 
query time, `bm25s` simply looks up the the per-keyword scores and sums them.
src: https://github.com/xhluca/bm25s

### Semantic search

Documents are embedded, and ranked by proximity (cosine similarity or inner product) in vector space. In this implementation, embeddings are 
generated with Hugging Face, and the vector database with FAISS.

### Reciprocal rank fusion

Combines multiple ranked lists into one by summing the reciprocals of each document's ranks (rewards high rankings on each list). In this 
implementation, a constant K is added in the denominator to prevent top-ranked results from dominating the ranking system.

### Pipeline

Retrieve relevant papers -> determine subtask -> execute subtask

## Installation:
`pip3 install -r requirements.txt`

## Usage:
```
from agent import Agent

agent = Agent()

query = "Can you summarize what's novel about this abstract?"

response = agent.run(query)

print(response)

```

## Data

The retriever pulls paper abstracts from the ai-arxiv2 dataset. 
More info at https://huggingface.co/datasets/jamescalam/ai-arxiv2
