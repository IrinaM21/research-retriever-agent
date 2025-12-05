from typing import List
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import bm25s

class Retriever:

  """
  Hybrid retriever combining:
  - BM25 keyword search (lexical)
  - FAISS dense retrieval (semantic)

  Uses Reciprocal Rank Fusion (RRF) to merge rankings.
  """
  
  def __init__(self, embed_model: str ="sentence-transformers/all-MiniLM-L6-v2", 
               chunk_size: int =100, chunk_overlap: int=20, top_k: int=3) -> None:
    """ 
    Initialize the hybrid retriever.

      Args:
          embed_model: HuggingFace embedding model name for semantic search.
          chunk_size: Character size of text chunks for FAISS embedding.
          chunk_overlap: Number of overlapping characters between chunks.
          top_k: Number of results to return per search method.
    """

    data = load_dataset("jamescalam/ai-arxiv2", split="train[:1%]")

    self.text = [x["summary"] for x in data]

    self.docs = [Document(page_content=t, metadata={"id": i}) 
             for i, t in enumerate(self.text)]

    self.id_to_text = {i: t for i, t in enumerate(self.text)}
    
    self.keyword_retriever = bm25s.BM25(corpus=self.text)
    
    self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
    )
    chunks = self.splitter.split_documents(self.docs)

    self.top_k = top_k

    self.embed = HuggingFaceEmbeddings(model_name=embed_model)

    self.vecdb = FAISS.from_documents(chunks, self.embed)

    self.rrf_k = 60   

  
  def keyword_search(self, query) -> List[int]:
    
    """
    Retrieve documents using BM25 keyword matching.

    Args:
        query: User query string.

    Returns:
        A list of top_k matching abstracts ranked by BM25 score (IDs).
    """

    tokenized_query = self.keyword_retriever.tokenize(query)
    keyword_res = self.keyword_retriever.retrieve(tokenized_query, k=self.top_k)
    return [hit[2] for hit in keyword_res] 
  
  def semantic_search(self, query) -> List[int]:
    """
    Retrieve documents using dense semantic similarity via FAISS.

    Args:
        query: User query string.

    Returns:
        A list of top_k most semantically similar abstract chunks (IDs).
    """
    
    semantic_res = self.vecdb.similarity_search_with_score(query, k=self.top_k)
    ids = [match.metadata["id"] for match, score in semantic_res]
    
    return list(dict.fromkeys(ids))[:self.top_k]

  
  def hybrid_search(self, query) -> List[str]:

    """
    Reciprocal Rank Fusion (RRF) to merge keyword and semantic rankings.

    Args:
        query: User query string.

    Returns:
     top_k docs, full text (strings).
    """
    K = self.rrf_k
    keyword_hits = self.keyword_search(query)
    semantic_hits = self.semantic_search(query)

    rrf_scores = {}
    for lst in [keyword_hits, semantic_hits]:
        for rank, item in enumerate(lst, start=1): 
                                                   
            if item not in rrf_scores:
                rrf_scores[item] = 0

            rrf_scores[item] += 1/(rank + K)

    
    sorted_items = sorted(rrf_scores, key=rrf_scores.get, reverse = True)

    return [self.id_to_text[i] for i in sorted_items[:self.top_k]]
