from retriever import Retriever
import textwrap
import logging
from llama_cpp import Llama
from tqdm import tqdm
import time

# Set up logging for traceability
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Agent:

  """
  LLM-based research agent with retrieval, routing, and synthesis.

  The agent retrieves relevant papers using a hybrid retriever,
  routes the user query to a subtask (creative or technical), and generates answers
  using the retrieved paper abstracts.

  """
  
  def __init__(self, llm_name:str, embed_name:str) -> None:

    """
    Initialize the agent.

    Loads:
        - Hybrid retriever
        - Tokenizer
        - LLM for routing and answering queries

    Logs progress for each component.

    Args:
        llm_name: Path to GGUF model file.
        embed_name: Embedding model name for retriever.
    """

    logger.info("Building agent...")

    self.retriever = Retriever(embed_model=embed_name)
    logger.info("Loaded retriever!")

    self.model = Llama(model_path=llm_name, n_threads=8, n_ctx=8192, n_gpu_layers=40, verbose=False)
                                                       
    logger.info("Loaded models!")
    
  def gen_from_prompt(self, prompt: str, **kwargs) -> str:
    """
    Args:
        prompt: Input prompt to send to the LLM.
        **kwargs: Generation args (temperature, max_tokens, etc.)

    Returns:
        Model-generated text.
    """

    # Default generation parameters, overridable by kwargs
    params = {
        "max_tokens": kwargs.pop("max_tokens", 512),
        "temperature": kwargs.pop("temperature", 0.7),
        "top_p": kwargs.pop("top_p", 0.9),
        "stop": kwargs.pop("stop", None),
    }

    output = self.model(
        prompt,
        **params,
    )

    return output["choices"][0]["text"].strip()
  
  
  def route_query(self, query: str) -> str:
    """
    Classify the user query into 'creative' or 'technical'.

    Routing prompt instructs the LLM to determine whether the user
    is asking:
        - Creative questions (suggesting research directions)
        - Technical questions (summarizing trends, explaining concepts)

    Args:
        query: The user's original request.

    Returns:
        Either "creative" or "technical".
    """

    # explicitly define role as a classifier to reduce creative drift causing responses not in [creative, technical]
    # list response options as individual words without punctuation to avoid generation of unexpected tokens
    prompt = f"""
            You are a classifier that determines the user's intent when asking about AI research.

            Examples:

            Query: Can you outline some future research directions for LLMs?
            Answer: creative

            Query: How have researchers tackled high computational cost in AI research?
            Answer: technical

            Query: What are the novel contributions in this abstract?
            Answer: technical

            Query: Suggest some open problems in computer vision.
            Answer: creative

            Query: {query}
            

            Respond with EXACTLY one word: creative or technical
            """
    
    logger.info("Routing request...")
      
    out = self.gen_from_prompt(
        prompt,
        max_tokens=2,
        temperature=0.0,          # low temp for classification
        stop=["\n"],
    )

    return out.strip().lower()

  def answer_creative(self, query: str) -> str:

    """
    Answer a creative question using relevant abstracts.

    Args:
        query: User query

    Returns:
        A 3–5 sentence response.
    """

    # clear role and response format definition (see route_query)
    prompt = f"""
            You are a research analyst. Answer in EXACTLY 1-3 sentences. Do NOT repeat or include any part of the prompt.

            Use the following papers as background context only. Do not quote or restate them.
            {self.relevant_papers}

            The user is asking a creative research question:
            {query}
            """
    
    return self.gen_from_prompt(
        prompt,
        max_tokens=200,
        temperature=0.7, # higher temp for creativity
        top_p=0.9,
        stop=["\n\n"],   # avoid rambling
    )
  
  def get_relevant_papers(self, query: str) -> None:
    """
    Retrieve relevant papers for the given query using the hybrid retriever.

    Args:
        query: User query string.
    """

    logger.info("Retrieving relevant papers...")
    self.relevant_papers = "\n\n".join(
        textwrap.shorten(p, width=500, placeholder="…") for p in tqdm(self.retriever.hybrid_search(query))
        )
    logger.info("Done!!!")
  
  def answer_technical(self, query: str) -> str:

    """
    Answer a technical question using relevant abstracts.

    Args:
        query: User query

    Returns:
        A 3–5 sentence response with citations for retrieved papers.
    """

    prompt = f"""
      You are a research analyst. Answer the user's technical question using the relevant papers provided.

      Nearest related papers:
      {self.relevant_papers}

      Query:
      {query}

      Answer in 3-5 sentences. Do NOT repeat or include any part of the prompt. Cite sources by number.
      """

    return self.gen_from_prompt(
        prompt,
        max_tokens=500,
        temperature=0.6, # slightly lower temp for technical accuracy
        top_p=0.9,
        stop=["\n\n"],
    )