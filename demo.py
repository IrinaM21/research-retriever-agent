from fastapi import FastAPI
from pydantic import BaseModel
from agent import Agent
import logging

# Set up logging for traceability
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize agent
agent = Agent(
    llm_name="./models/mistral-7b-instruct-v0.3-q4_k_m.gguf",
    embed_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Pydantic model for request payload
class Query(BaseModel):
    query: str

# Endpoint for generating responses
@app.post("/generate")
def generate(payload: Query) -> dict:

    """
    Full agent pipeline:
    1. Retrieve relevant papers
    2. Route the query to 'creative' or 'technical'
    3. Execute the chosen reasoning task

    Args:
        query: The user query.

    Returns:
        A generated response tailored to the detected task type.
    """

    # Route the query using Agent's own LLM
    subtask = agent.route_query(payload.query)

    # Retrieve relevant papers
    agent.get_relevant_papers(payload.query)
    
    # Generate answer based on routing
    if subtask == "creative":
        text = agent.answer_creative(payload.query)
    elif subtask == "technical":
        text = agent.answer_technical(payload.query)
    else:
        logger.warning("Your query is outside my expertise. I'll try to help anyway! Defaulting to creative response.")
        text = agent.answer_creative(payload.query)

    return {"type": subtask, "text": text, "src": agent.relevant_papers}