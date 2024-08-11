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

class SummaryTool(BaseModel):
    input_text: str = Field(..., title="Input Text", description="The input text to summarize")
    metadata: Dict[str, Any] = Field({}, title="Metadata", description="Metadata to pass to the model")

@tool(args_schema=SummaryTool)
def summarize_text(research_text: str, metadata: dict) -> str:
    """
    Summarize the research text for the article
    """
    model = ChatOpenAI()
    
    fine_grained_prompt = hub.pull("fine_grained_prompt_summarize")

    model_chain = fine_grained_prompt | model 

    response = model.invoke({"input": research_text})

    return response.content

def summarize_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the agent on the given state
    """
    messages = state['messages']
    response = summarize_text(messages)
    return response
