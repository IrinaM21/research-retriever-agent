from retriever import Retriever
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import textwrap
import logging

logger = logging.getLogger(__name__)

class Agent:

  """
  LLM-based research agent with retrieval, routing, and synthesis.

  The agent retrieves relevant papers using a hybrid retriever,
  routes the user query to a subtask (novelty analysis or research
  direction generation), and generates answers using a small LLM.

  """

  def __init__(self) -> None:

    """
    Initialize the agent.

    Loads:
        - Hybrid retriever
        - Tokenizer
        - Small LLM (Phi-3.5-mini-instruct)

    Logs progress for each component.
    """

    logger.info("Building agent...")

    model_name = "microsoft/Phi-3.5-mini-instruct"

    self.retriever = Retriever(embed_model="sentence-transformers/all-MiniLM-L6-v2")
    logger.info("Loaded retriever!")
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info("Loaded tokenizer!")

    self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name,
                                                      device_map="auto",
                                                      torch_dtype=torch.bfloat16) 
    self.model.eval()
    logger.info("Loaded models!")
    
    
  def gen_from_prompt(self, prompt: str, **kwargs) -> str:

    """
    Generate text from the underlying language model.

    Args:
        prompt: Input prompt to send to the LLM.
        **kwargs: Additional generation arguments (e.g., temperature).

    Returns:
        Model-generated text decoded into a string.
    """

    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
    with torch.no_grad():
        outputs = self.model.generate(**inputs, **kwargs)
    return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
  
  
  def route_query(self, query: str) -> str:
    """
    Classify the user query into 'novelty' or 'direction'.

    A routing prompt instructs the LLM to determine whether the user
    is asking for:
        - Novelty analysis (understanding or summarizing contributions)
        - Research direction generation (idea creation)

    Args:
        query: The user's original request.

    Returns:
        Either "novelty" or "direction".
    """

    prompt = f"""
      Determine if the user would needs help generating new ideas or synthesizing existing information.
      Examples:
      1. Query: Can you summarize what's novel about the technique in this abstract? -> novelty
      2. Query: Give me 2-3 new research directions based on this abstract. -> direction
      3. Query: What related work is this paper building on and how? -> novelty
      4. Query: I'd like to learn more about the concepts in this abstract. What should I read? -> novelty
      5. Query: I'd like to contribute to the concepts in this abstract. Help me brainstorm. -> direction
      Query: {query}
      Answer: Respond in one word, either "novelty" or "direction."
      """
    
    logger.info("Routing request...")
      
    return self.gen_from_prompt(
        prompt,
        max_length=2000,
        do_sample=False,
        top_k=50
        ).strip().split()[-1].lower()

  def novelty_detect(self, query: str) -> str:

    """
    Analyze novelty by comparing the abstract to retrieved papers.

    Generates a short explanation describing the core innovations of
    the abstract relative to retrieved related work.

    Args:
        query: User query containing or referencing a research abstract.

    Returns:
        A 3–5 sentence novelty analysis with citations to retrieved papers.
    """

    prompt = f"""
      Compare the abstract in the prompt to the nearest related papers and explain what novelty it introduces. 
      Nearest related papers:
      {self.relevant_papers}

      Query: {query}

      Respond in 3–5 sentences. Cite sources by number.
      """
    
    return self.gen_from_prompt(
          prompt,
          max_length=2000,
          do_sample=False,
          top_k=50,
          temperature=0.1
      )
  
  def find_directions(self, query: str) -> str:

    """
    Suggest future research directions based on the abstract.

    Uses retrieved related papers and the query abstract to propose
    plausible extensions, follow-up experiments, or open research paths.

    Args:
        query: User query containing or referencing a research abstract.

    Returns:
        A 3–5 sentence set of research direction suggestions with citations.
    """

    prompt = f"""
      Use the abstract in the query and relevant papers to suggest some future research directions.
      Nearest related papers:
      {self.relevant_papers}

      Query: {query}

      Respond in 3–5 sentences. Cite sources by number.
      """

    return self.gen_from_prompt(
          prompt,
          max_length=2000,
          do_sample=True,
          top_k=50,
          temperature=0.7,
          top_p=0.9
      )
  
  def run(self, query: str) -> str:
    """
    Full agent pipeline:
    1. Retrieve relevant papers
    2. Route the query to 'novelty' or 'direction'
    3. Execute the chosen reasoning task

    Args:
        query: The user query.

    Returns:
        A generated response tailored to the detected task type.
    """


    logger.info("Retrieving relevant papers...")
    self.relevant_papers = "\n\n".join(
        textwrap.shorten(p, width=500, placeholder="…") for p in self.retriever.hybrid_search(query)
        )
    logger.info("Done!!!")

    subtask = self.route_query(query)

    self.trace = {
      "query": query,
      "retrieved_papers": self.relevant_papers,
      "subtask": subtask,
    }

    if subtask not in ["novelty", "direction"]:
      logger.warning(f"Router returned unexpected mode: {subtask}. Defaulting to novelty.")
      subtask = "novelty"
    
    if "novel" in subtask:
      return self.novelty_detect(query)

    return self.find_directions(query)
