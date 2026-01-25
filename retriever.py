from typing import List
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
import bm25s
import numpy as np
import logging
from tqdm import tqdm
import time

# Set up logging for traceability
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Retriever:

  """
  Hybrid retriever combining:
  - BM25 keyword search (lexical)
  - FAISS dense retrieval (semantic)

  Uses Reciprocal Rank Fusion (RRF) to merge rankings.
  """
  
  def __init__(self, embed_model: str ="sentence-transformers/all-MiniLM-L6-v2", top_k: int=3) -> None:

    """ 
    Initialize the hybrid retriever.

      Args:
          embed_model: Embedding model for semantic search.
          top_k: Number of results to return per search method.
    """

    data = load_dataset("jamescalam/ai-arxiv2", split="train")
    self.texts = data["summary"]

    self.top_k = top_k # number of results to return per method
    self.rrf_k = 60 # RRF parameter to dampen lower ranks
    
    self.embed = SentenceTransformer(embed_model)
    logger.info("Generating embeddings for FAISS index...")
    embeddings = self.embed.encode(self.texts)
    embeddings = np.array(embeddings).astype("float32")

    logger.info("Building BM25 index...")
    self.keyword_retriever = bm25s.BM25()
    self.keyword_retriever.index(bm25s.tokenize(self.texts))

    logger.info("Building FAISS index...")
    dim = embeddings.shape[1]
    self.index = faiss.IndexFlatL2(dim)
    self.index.add(embeddings)  

  
  def keyword_search(self, query) -> List[int]:
    
    """
    Retrieve documents using BM25 keyword matching.

    Args:
        query: User query string.

    Returns:
        A list of top_k matching abstracts ranked by BM25 score (IDs).
    """

    tokenized_query = bm25s.tokenize(query)
    keyword_res = self.keyword_retriever.retrieve(tokenized_query, k=self.top_k) # list of (doc_id, score) tuples
    return (keyword_res[0][0]).tolist() # extract doc IDs only
  
  def semantic_search(self, query) -> List[int]:
    """
    Retrieve documents using dense semantic similarity via FAISS.

    Args:
        query: User query string.

    Returns:
        A list of top_k most semantically similar abstract chunks (IDs).
    """

    q_emb = self.embed.encode_query(query)
    q_emb = np.array([q_emb]).astype("float32")

    dists, idx = self.index.search(q_emb, k=self.top_k) # idx shape: (1, top_k)
    print(idx)  # for debugging
    return idx[0].astype(int).tolist() # extract doc IDs only

  
  def hybrid_search(self, query) -> List[str]:

    """
    Reciprocal Rank Fusion (RRF) to merge keyword and semantic rankings.

    Args:
        query: User query string.

    Returns:
     top_k docs, full text (strings).
    """
    keyword_hits = self.keyword_search(query)
    semantic_hits = self.semantic_search(query)

    # RRF scoring
    rrf_scores = {}
    # Iterate over both lists of hits
    for lst in tqdm([keyword_hits, semantic_hits]):
        for rank, item in enumerate(lst, start=1): 

            # Initialize score if not present                                       
            if item not in rrf_scores:
                rrf_scores[item] = 0

            # Update RRF score
            rrf_scores[item] += 1/(rank + self.rrf_k)

    # Sort items by RRF score in descending order
    sorted_items = sorted(rrf_scores, key=rrf_scores.get, reverse = True)

    # Return top_k documents' full text
    return [self.texts[i] for i in sorted_items[:self.top_k]]
