from langchain import hub
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from typing import List, Dict, Any
from pydantic.v1 import BaseModel, Field
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.callbacks import StdOutCallbackHandler
from langgraph.graph import StateGraph
from dotenv import load_dotenv
import os
import sys
from .llm_node import get_model

class FetchResearchContentTool(BaseModel):
    paper_url: str = Field(..., title="URL", description="The URL path to fetch the research content from")
    metadata: Dict[str, Any] = Field({}, title="Metadata", description="Metadata to pass to the model")

@tool(args_schema=FetchResearchContentTool)
def fetch_research_content(paper_url: str, metadata: dict) -> str:
    """
    Fetch the research content from the path
    """

    loader = WebBaseLoader()
    research_text = loader.load(paper_url)

    return research_text

def fetch_content_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the agent on the given state
    """
    messages = state['messages']

    llm = get_model()

    llm_chain = llm.bind_tools([fetch_research_content])

    response = llm_chain.invoke(messages)

    return response

    
    
