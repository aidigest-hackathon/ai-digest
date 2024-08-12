from langchain import hub
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from typing import List, Dict, Any
from pydantic.v1 import BaseModel, Field
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.callbacks import StdOutCallbackHandler
from langgraph.graph import StateGraph
from dotenv import load_dotenv
import os
import sys

def get_model()->ChatOpenAI:
    model = ChatOpenAI()

def get_model_with_tools(model: ChatOpenAI, tools: List[tool]) -> Any:
    return model.bind_tools(tools)

def get_model_chain(model:ChatOpenAI, prompt) -> Any:
    """Load the model chain that can be invoked with user input"""
    prompt = hub.pull("basic_research_curation_prompt")
    return prompt | model
